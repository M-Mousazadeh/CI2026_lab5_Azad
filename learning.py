"""Solution 2: a learning classifier system (Michigan style) and the GA that simplifies what it learns.

The population starts empty and the agent does not know the map. Three components:
  * discovery, covering: when the match set has no classifier for some action, one is created with
    the whole message list as condition and an optimistic strength of 1000, so untried actions win
    the auction and get tried;
  * performance, auction: the strongest matching classifier wins (while learning, with probability
    `explore` a random matching classifier wins instead);
  * reinforcement, discounted bucket brigade: the winner pays its predecessor, which collects the
    reward of its own action plus the discounted best bid of the classifiers it set the stage for:
        S_prev <- S_prev + beta * (r_prev + gamma * best_bid - S_prev)
    and the last winner of an episode collects its reward: S <- S + beta * (r - S).
After learning, a GA (compact) generalises and prunes the classifiers that the greedy run uses.

Memory. With messages only, the learner never solves the cave (see the ablation in lab5.py): the
same message list appears on different squares, so a death on one square poisons the strength of a
classifier that is safe on another. So the learner gets the agent's own bookkeeping (DeadReckoning)
as internal messages. Messages-only learning (LearningCS with tags > 0) is kept for that ablation.
"""
import random
from dataclasses import dataclass

from cave import DELTA, EFFECTORS, HEADINGS
from classifiers import MAX_STEPS, Classifier, posted, run_episode


class DeadReckoning:
    """The agent's own bookkeeping. The agent cannot perceive its square, but it knows which effector
    it executed and whether it bumped into a wall. From these it keeps its position, its heading and
    whether it carries the gold, and adds them to every message list as internal messages, e.g.
    {x2, y1, facing_north, no_gold}. It never reads the cave."""

    def reset(self):
        self.x, self.y, self.heading, self.has_gold = 1, 1, 'east', False

    def messages(self):
        return frozenset({f'x{self.x}', f'y{self.y}', f'facing_{self.heading}',
                          'has_gold' if self.has_gold else 'no_gold'})

    def update(self, effector, before, after):
        """effector: what the agent just did; before/after: the sensor messages around that action."""
        if effector == 'fwd' and 'bump' not in after:
            dx, dy = DELTA[self.heading]
            self.x, self.y = self.x + dx, self.y + dy
        elif effector in ('turn_l', 'turn_r'):
            turn = 1 if effector == 'turn_l' else -1
            self.heading = HEADINGS[(HEADINGS.index(self.heading) + turn) % 4]
        elif effector == 'grab' and 'glitter' in before:
            self.has_gold = True


@dataclass(eq=False)
class LearnedClassifier(Classifier):
    stamp: int = 0                  # last time the online GA ran in this classifier's niche


class LearningCS:
    """memory: DeadReckoning() or None.
    tags: 0, or K > 0 for messages-only learning: every action then also posts one of the internal
          messages m0..m(K-1), which is the agent's only memory (the first list holds m0).
    online_ga: run a strength-based niche GA while learning (ablation only)."""

    def __init__(self, memory=None, tags=0, seed=0, beta=0.5, gamma=0.9, explore=0.2,
                 initial_strength=1000.0, online_ga=False, ga_period=25, mutation=0.1, max_population=1500):
        self.memory = memory
        self.rng = random.Random(seed)
        self.beta, self.gamma, self.explore, self.initial_strength = beta, gamma, explore, initial_strength
        if tags:
            names = [f'm{i}' for i in range(tags)]
            self.actions = [frozenset({e, m}) for e in EFFECTORS for m in names]
            self.initial = frozenset({'m0'})
        else:
            self.actions = [frozenset({e}) for e in EFFECTORS]
            self.initial = frozenset()
        self.online_ga, self.ga_period = online_ga, ga_period
        self.mutation, self.max_population = mutation, max_population
        self.rules = []                                     # the population starts empty
        self.t = 0

    def match(self, msgs):
        """The match set, completed by covering."""
        match = [c for c in self.rules if c.matches(msgs)]
        present = {c.action for c in match}
        for action in self.actions:
            if action not in present:
                c = LearnedClassifier(msgs, action, self.initial_strength, stamp=self.t)
                self.rules.append(c)
                match.append(c)
        return match

    def episode(self, cave):
        """One learning episode. Returns its score."""
        sensors, internal = cave.reset(), self.initial
        if self.memory is not None:
            self.memory.reset()
        previous, score = None, 0
        for _ in range(MAX_STEPS):
            msgs = sensors | internal | (self.memory.messages() if self.memory is not None else frozenset())
            match = self.match(msgs)
            best = max(c.strength for c in match)
            if self.rng.random() < self.explore:
                winner = self.rng.choice(match)
            else:
                winner = self.rng.choice([c for c in match if c.strength == best])
            if previous is not None:                        # bucket brigade: pay the previous winner
                prev, prev_reward = previous
                prev.strength += self.beta * (prev_reward + self.gamma * best - prev.strength)
            if self.online_ga:
                self.niche_ga(match, winner, msgs)
            self.t += 1
            new_sensors, reward, done = cave.step(winner.effector)
            if self.memory is not None:
                self.memory.update(winner.effector, sensors, new_sensors)
            sensors, internal, score = new_sensors, posted(match, winner), score + reward
            if done:                                        # death or climb: nobody comes after
                winner.strength += self.beta * (reward - winner.strength)
                break
            previous = (winner, reward)
        return score                                        # cut after MAX_STEPS: not a terminal state

    def test(self, cave):
        """Greedy run with the learned classifiers: no learning, no covering, no exploration."""
        return run_episode(self.rules, cave, memory=self.memory, initial=self.initial)

    def niche_ga(self, match, winner, msgs):
        """Online GA (ablation). Every ga_period steps per niche, two strong classifiers that advocate
        the winning action make a child: uniform crossover and mutation of their conditions (kept
        inside the current message list), and the child starts with the parents' mean strength."""
        niche = [c for c in match if c.action == winner.action]
        if self.t - sum(c.stamp for c in niche) / len(niche) < self.ga_period:
            return
        for c in niche:
            c.stamp = self.t

        def parent():
            return max(self.rng.sample(niche, min(3, len(niche))), key=lambda c: c.strength)

        p1, p2 = parent(), parent()
        names = sorted(msgs)                                # a fixed order keeps runs reproducible
        condition = frozenset(m for m in names if m in (p1 if self.rng.random() < 0.5 else p2).condition)
        condition = frozenset(m for m in names if (m in condition) != (self.rng.random() < self.mutation))
        action = p1.action if self.rng.random() >= self.mutation else self.rng.choice(self.actions)
        if any(c.condition == condition and c.action == action for c in self.rules):
            return
        self.rules.append(LearnedClassifier(condition, action, (p1.strength + p2.strength) / 2, stamp=self.t))
        while len(self.rules) > self.max_population:        # delete the weakest of 5 random classifiers
            self.rules.remove(min(self.rng.sample(self.rules, 5), key=lambda c: c.strength))


def compact(rules, cave, memory=None, seed=0, population=40, generations=150, crossover=0.7, tournament=3):
    """A GA that simplifies a list of learned classifiers without losing score.
    Genome: one bit per classifier (keep it?) and one bit per message of its condition (keep the message?).
    Fitness, compared lexicographically: (score of a greedy run, -classifiers, -condition messages).
    The simplified classifiers keep their learned strengths, which still decide every auction.
    Returns (simplified classifiers, fitness)."""
    rng = random.Random(seed)
    slots = [(c, sorted(c.condition)) for c in rules]
    length = sum(1 + len(names) for _, names in slots)

    def decode(genome):
        kept, i = [], 0
        for c, names in slots:
            keep, bits = genome[i], genome[i + 1:i + 1 + len(names)]
            i += 1 + len(names)
            if keep:
                kept.append(Classifier(frozenset(n for n, b in zip(names, bits) if b), c.action, c.strength))
        return kept

    cache = {}

    def fitness(genome):
        key = tuple(genome)
        if key not in cache:
            kept = decode(genome)
            score, _ = run_episode(kept, cave, memory=memory)
            cache[key] = (score, -len(kept), -sum(len(c.condition) for c in kept))
        return cache[key]

    def select(pop):
        return max(rng.sample(pop, tournament), key=fitness)

    pop = [[1] * length] + [[0 if rng.random() < 0.2 else 1 for _ in range(length)] for _ in range(population - 1)]
    for _ in range(generations):
        children = [max(pop, key=fitness)]                  # elitism
        while len(children) < population:
            a, b = select(pop), select(pop)
            child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)] if rng.random() < crossover else list(a)
            children.append([1 - x if rng.random() < 1 / length else x for x in child])
        pop = children
    best = max(pop, key=fitness)
    return decode(best), fitness(best)

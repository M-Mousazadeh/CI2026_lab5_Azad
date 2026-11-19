"""Classifiers, the message-list cycle (the performance system) and the hand-written classifier set.

A classifier is a pair of sets (slides): it matches when its condition is a subset of the message
list, and the empty condition matches everything. Its action holds one effector message (executed by
the cave) and/or internal messages, which go to the next message list.

One step of the cycle:
  1. message list = sensor messages + internal messages posted at the previous step
                    (+ the messages of the agent's bookkeeping, if it has one: see learning.py)
  2. match set    = every classifier whose condition is a subset of the message list
  3. every matching classifier without an effector ("internal classifier") fires: it only posts messages
  4. among the matching classifiers with an effector, the strongest one wins the auction: its effector
     acts on the cave and its internal messages are posted
  5. the posted messages form the internal part of the next message list; all other messages vanish
"""
from dataclasses import dataclass
from typing import NamedTuple

from cave import EFFECTORS, SENSORS

MAX_STEPS = 100                     # an episode is cut after 100 actions (not a terminal state)


@dataclass(eq=False)                # eq=False: two classifiers are the same only if they are the same object
class Classifier:
    condition: frozenset
    action: frozenset
    strength: float = 0.0

    def matches(self, msgs):
        return self.condition <= msgs

    @property
    def effector(self):
        """The effector message of the action, or None for an internal classifier."""
        found = [m for m in self.action if m in EFFECTORS]
        if len(found) > 1:
            raise ValueError(f'{self}: an action can drive only one effector')
        return found[0] if found else None

    @property
    def internal(self):
        """The internal messages that the classifier posts to the next list."""
        return frozenset(m for m in self.action if m not in EFFECTORS)

    def __str__(self):
        return f'{show(self.condition)} -> {show(self.action)}'


def rule(condition, action, strength=0.0):
    """rule('has_gold at_start', 'climb') -> Classifier({has_gold, at_start}, {climb})"""
    return Classifier(frozenset(condition.split()), frozenset(action.split()), strength)


def show(msgs):
    """Messages in a fixed order: sensors, then effectors, then internal messages (sorted)."""
    order = {m: i for i, m in enumerate(SENSORS + EFFECTORS)}
    return '{' + ', '.join(sorted(msgs, key=lambda m: (order.get(m, len(order)), m))) + '}'


def auction(match, rng=None):
    """The winner: the matching classifier with an effector and the highest strength.
    Ties go to the first one in the list, or to a random one if a random generator is given."""
    bidders = [c for c in match if c.effector is not None]
    if not bidders:
        return None
    best = max(c.strength for c in bidders)
    tied = [c for c in bidders if c.strength == best]
    return tied[0] if rng is None else rng.choice(tied)


def posted(match, winner):
    """The internal messages for the next list: the winner's, plus those of every matching internal classifier."""
    msgs = set(winner.internal)
    for c in match:
        if c.effector is None:
            msgs |= c.action
    return frozenset(msgs)


class Step(NamedTuple):
    t: int
    square: tuple                   # where the agent really was (for the trace only: the agent never sees it)
    heading: str
    msgs: frozenset                 # the message list
    winner: Classifier
    posted: frozenset               # internal messages for the next list
    reward: int


def run_episode(rules, cave, memory=None, initial=frozenset(), max_steps=MAX_STEPS, rng=None):
    """Run one episode with a fixed set of classifiers (no learning). Returns (score, trace).
    memory: optional bookkeeping that adds its own messages to every list (learning.DeadReckoning).
    initial: internal messages of the first list."""
    sensors, internal = cave.reset(), frozenset(initial)
    if memory is not None:
        memory.reset()
    score, trace = 0, []
    for t in range(1, max_steps + 1):
        msgs = sensors | internal | (memory.messages() if memory is not None else frozenset())
        match = [c for c in rules if c.matches(msgs)]
        winner = auction(match, rng)
        if winner is None:                                  # no classifier can act: the agent stops
            break
        square, heading = cave.pos, cave.heading
        new_sensors, reward, done = cave.step(winner.effector)
        if memory is not None:
            memory.update(winner.effector, sensors, new_sensors)
        internal = posted(match, winner)
        trace.append(Step(t, square, heading, msgs, winner, internal, reward))
        sensors, score = new_sensors, score + reward
        if done:
            break
    return score, trace


# ---- Solution 1: hand-written classifiers that use messages only ------------------------------------
# The agent never knows where it is. Its only memory is the internal messages its classifiers post:
#   has_gold  "I carry the gold": posted by the grab classifier and kept alive by the last classifier
#   m1, m2    "I have just turned once / twice": they let the same sensors lead to different actions
# All strengths are equal, so the first matching classifier in the list wins: exceptions come before
# the default {} -> fwd (a default hierarchy).
HAND_WRITTEN = [
    rule('glitter', 'grab has_gold'),         # the gold is here: take it and remember it
    rule('has_gold at_start', 'climb'),       # back home with the gold: leave
    rule('m2', 'fwd'),                        # U-turn completed: walk back down column 2
    rule('has_gold stench m1', 'turn_l m2'),  # (2,3) with the gold, turned once: turn again
    rule('has_gold stench', 'turn_l m1'),     # (2,3) with the gold: start the U-turn (W is the Wumpus, E a pit)
    rule('breeze m1', 'fwd'),                 # (2,1) after the turn: go on (north, or west on the way back)
    rule('has_gold breeze', 'turn_r m1'),     # (2,1) coming back south: face west, towards the start
    rule('breeze', 'turn_l m1'),              # (2,1) coming from the start: a pit is ahead, face north
    rule('', 'fwd'),                          # default: walk forward
    rule('has_gold', 'has_gold'),             # internal classifier: keep remembering the gold
]

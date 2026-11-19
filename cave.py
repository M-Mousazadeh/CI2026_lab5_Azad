"""The 4x4 Wumpus World cave of the Lab 5 slides (AIMA, Figure 7.2).

Squares are (x, y) with (1, 1) at the bottom left. The agent starts at (1, 1) facing east.
The cave talks to the agent only through messages (Python strings):
  * sensor messages, posted by the cave:     breeze, stench, glitter, bump, scream, at_start
  * effector messages, executed by the cave: fwd, turn_l, turn_r, grab, shoot, climb

Rewards (AIMA): every action costs 1, leaving the cave with the gold gives +1000, walking into
a pit or into the live Wumpus gives -1000 and ends the episode, and using the arrow costs 10 more.

Choices the slides leave open:
  * climb at (1, 1) always ends the episode, but pays +1000 only if the agent carries the gold;
    climb anywhere else does nothing;
  * a dead Wumpus no longer smells, and it is harmless;
  * bump and scream are perceived only right after the action that caused them.
"""

SENSORS = ('breeze', 'stench', 'glitter', 'bump', 'scream', 'at_start')
EFFECTORS = ('fwd', 'turn_l', 'turn_r', 'grab', 'shoot', 'climb')
HEADINGS = ('east', 'north', 'west', 'south')             # turn_l = next one, turn_r = previous one
DELTA = {'east': (1, 0), 'north': (0, 1), 'west': (-1, 0), 'south': (0, -1)}
START = (1, 1)

STEP_COST, ARROW_COST, DEATH, GOLD = -1, -10, -1000, 1000


class Cave:
    """The environment. Only reset() and step() are meant for the agent: they return sensor messages.
    The attributes (pos, heading, has_gold, ...) are read only by traces, plots and checks."""

    def __init__(self, wumpus=(1, 3), gold=(2, 3), pits=((3, 1), (3, 3), (4, 4)), size=4):
        self.size, self.wumpus, self.gold_square, self.pits = size, wumpus, gold, frozenset(pits)
        for square in (wumpus, gold, *pits):
            if not self.inside(square) or square == START:
                raise ValueError(f'{square} must be inside the cave and different from the start')
        self.reset()

    def inside(self, square):
        return 1 <= square[0] <= self.size and 1 <= square[1] <= self.size

    def neighbours(self, square):
        x, y = square
        return [(x + dx, y + dy) for dx, dy in DELTA.values() if self.inside((x + dx, y + dy))]

    def ahead(self):
        dx, dy = DELTA[self.heading]
        return self.pos[0] + dx, self.pos[1] + dy

    def reset(self):
        self.pos, self.heading = START, 'east'
        self.gold = self.gold_square                        # None once the agent has grabbed it
        self.has_gold, self.arrow, self.wumpus_alive = False, True, True
        self.bump = self.scream = False
        self.done = self.dead = self.escaped = False
        return self.sensors()

    def sensors(self):
        """The sensor messages for the agent's current square."""
        near = self.neighbours(self.pos)
        msgs = set()
        if self.pits.intersection(near):
            msgs.add('breeze')
        if self.wumpus_alive and (self.wumpus == self.pos or self.wumpus in near):
            msgs.add('stench')
        if self.gold == self.pos:
            msgs.add('glitter')
        if self.bump:
            msgs.add('bump')
        if self.scream:
            msgs.add('scream')
        if self.pos == START:
            msgs.add('at_start')
        return frozenset(msgs)

    def step(self, action):
        """Execute one effector message. Returns (sensor messages, reward, episode over)."""
        if self.done:
            raise RuntimeError('the episode is over: call reset()')
        if action not in EFFECTORS:
            raise ValueError(f'unknown effector {action!r}')
        self.bump = self.scream = False
        reward = STEP_COST
        if action == 'fwd':
            square = self.ahead()
            if not self.inside(square):
                self.bump = True                            # a wall: the agent does not move
            else:
                self.pos = square
                if square in self.pits or (self.wumpus_alive and square == self.wumpus):
                    reward += DEATH
                    self.done = self.dead = True
        elif action in ('turn_l', 'turn_r'):
            turn = 1 if action == 'turn_l' else -1
            self.heading = HEADINGS[(HEADINGS.index(self.heading) + turn) % 4]
        elif action == 'grab':
            if self.gold == self.pos:
                self.gold, self.has_gold = None, True
        elif action == 'shoot':
            if self.arrow:
                self.arrow = False
                reward += ARROW_COST
                dx, dy = DELTA[self.heading]
                square = (self.pos[0] + dx, self.pos[1] + dy)
                while self.inside(square):                  # the arrow flies until it hits a wall
                    if self.wumpus_alive and square == self.wumpus:
                        self.wumpus_alive, self.scream = False, True
                        break
                    square = (square[0] + dx, square[1] + dy)
        elif action == 'climb':
            if self.pos == START:
                self.done = self.escaped = True
                if self.has_gold:
                    reward += GOLD
        return self.sensors(), reward, self.done

    def percept_map(self):
        """The cave as in the slide, top row first: what the agent perceives on each square."""
        rows = []
        for y in range(self.size, 0, -1):
            cells = []
            for x in range(1, self.size + 1):
                square = (x, y)
                if square in self.pits:
                    cells.append('PIT')
                elif square == self.wumpus:
                    cells.append('WUMPUS')
                else:
                    near = self.neighbours(square)
                    names = []
                    if self.pits.intersection(near):
                        names.append('breeze')
                    if self.wumpus in near:
                        names.append('stench')
                    if square == self.gold_square:
                        names.append('glitter')
                    cells.append(' '.join(names) or '-')
            rows.append(cells)
        return rows

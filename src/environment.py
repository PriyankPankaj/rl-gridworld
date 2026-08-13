import numpy as np
import random


class GridWorld:
    """
    A configurable NxN GridWorld with optional stochastic ("slippery")
    transitions, following the same design as OpenAI Gym's FrozenLake.

    - State: (row, col), flattened to a single index
    - Actions: up, down, left, right
    - Reward function: -1 per step, +10 at goal, -10 at any hazard cell
    - Terminal conditions: reaching the goal, or falling into a hazard
    - Stochasticity: with probability (1 - slip_prob), the agent's chosen
      action succeeds; otherwise it "slips" into one of the two directions
      perpendicular to the intended one (e.g. intending "up" may result in
      "left" or "right" instead), matching classic slippery-grid RL setups.
    """

    ACTIONS = ["up", "down", "left", "right"]
    ACTION_DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    PERPENDICULAR = {
        "up": ["left", "right"],
        "down": ["left", "right"],
        "left": ["up", "down"],
        "right": ["up", "down"],
    }

    def __init__(self, size=10, start=(0, 0), goal=None, hazards=None, slip_prob=0.1):
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.hazards = hazards if hazards is not None else self._default_hazards(size)
        self.slip_prob = slip_prob
        self.n_states = size * size
        self.n_actions = len(self.ACTIONS)
        self.state = start

    def _default_hazards(self, size):
        # Scatter hazards proportionally to grid size, avoiding start/goal corners
        random.seed(42)  # fixed seed so the layout is reproducible across runs
        n_hazards = max(3, (size * size) // 20)
        hazards = set()
        while len(hazards) < n_hazards:
            r, c = random.randint(0, size - 1), random.randint(0, size - 1)
            if (r, c) not in [(0, 0), (size - 1, size - 1)]:
                hazards.add((r, c))
        random.seed()  # reset to non-deterministic for actual training randomness
        return list(hazards)

    def reset(self):
        self.state = self.start
        return self._to_index(self.state)

    def _to_index(self, pos):
        return pos[0] * self.size + pos[1]

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def _resolve_action(self, action):
        """Applies slip probability: returns the actual action taken."""
        if random.random() < self.slip_prob:
            return random.choice(self.PERPENDICULAR[action])
        return action

    def step(self, action_idx):
        intended_action = self.ACTIONS[action_idx]
        actual_action = self._resolve_action(intended_action)
        dr, dc = self.ACTION_DELTAS[actual_action]
        r, c = self.state
        new_pos = (r + dr, c + dc)

        if not self._in_bounds(new_pos):
            new_pos = self.state

        self.state = new_pos

        if new_pos == self.goal:
            return self._to_index(new_pos), 10.0, True
        elif new_pos in self.hazards:
            return self._to_index(new_pos), -10.0, True
        else:
            return self._to_index(new_pos), -1.0, False

    def random_action(self):
        return random.randint(0, self.n_actions - 1)
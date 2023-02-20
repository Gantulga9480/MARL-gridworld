from Game import Game
from Game import core
import numpy as np
import csv
import copy


OBJECTS = [
    {'color': (0, 0, 0), 'id': 0},        # Empty
    {'color': (125, 125, 125), 'id': 1},  # Obstacle
    {'color': (255, 0, 0), 'id': 2},      # Hole
    {'color': (255, 255, 0), 'id': 3},    # Agent
    {'color': (0, 255, 0), 'id': 4}       # Goal
]
E = OBJECTS[0]['id']  # Empty
W = OBJECTS[1]['id']  # Wall
H = OBJECTS[2]['id']  # Hole
A = OBJECTS[3]['id']  # Agent
G = OBJECTS[4]['id']  # Goal

# Moves
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Move checks
MOVABLE = 0
INPLACE = 1
FALLING = 2
TERMINATE = 3

# Rewards
REWARDS = [
    0,   # Empty move
    0,   # Stayed inplace
    -1,   # Fell through hole or time expired
    1    # Found goal
]

EPISODE_TERMINATING_POLICY = [
    False,  # Empty move - RESUME
    False,  # Stayed inplace - RESUME
    True,   # Fell through hole - TERMINATE
    True    # Found goal - TERMINATE
]


class GridEnv(Game):

    def __init__(self, env_file: str = None) -> None:
        """
        @param:
        env_file: str (default None) Load environment config from file.
        """
        super().__init__()
        self.box_size = 80
        self.fps = 30
        self.window_flags = 0
        self.initial_agent_location = None
        self.initial_goal_location = None
        self.agent_location = None
        self.initial_board = None
        self.board = None
        self.model = None
        self.over = False
        self.step_count = 0

        if env_file:
            self.load_env(env_file)
        else:
            self.initial_board = np.array([[E, E, E, E, G],
                                           [E, E, E, E, E],
                                           [A, E, E, E, E]])
            self.initial_agent_location = [2, 0]
            self.initial_goal_location = [0, 4]
        shape = self.initial_board.shape
        self.size = (shape[1] * self.box_size, shape[0] * self.box_size)
        self.max_step = (shape[1] - 1) * (shape[0] - 1)
        self.set_window()
        self.reset()

    @property
    def state_space_size(self):
        return self.initial_board.shape

    @property
    def action_space_size(self):
        return 4

    @property
    def observation_size(self):
        return 6

    def reset(self, pos=None):
        self.over = False
        self.step_count = 0
        if pos:
            y = pos[0]
            x = pos[1]
            if (self.initial_board[y, x] == E):
                self.agent_location = [y, x]
            else:
                raise ValueError()
        else:
            shape = self.initial_board.shape
            while True:
                x = np.random.randint(1, shape[1] - 1)
                y = np.random.randint(1, shape[0] - 1)
                if (self.initial_board[y, x] == E):
                    self.agent_location = [y, x]
                    break
        self.board = copy.deepcopy(self.initial_board)
        return self.get_state()

    def step(self, action):
        res = self._check_action(action)
        # Valid move
        if res != INPLACE:
            self._move(action)
        if self.step_count % self.max_step == 0:
            res = FALLING
        self.over = EPISODE_TERMINATING_POLICY[res]
        return self.get_state(), REWARDS[res], self.over

    def get_state(self):
        return self.get_state_dqn()

    def get_state_dqn(self):
        up = self.board[self.agent_location[0] - 1, self.agent_location[1]]
        down = self.board[self.agent_location[0] + 1, self.agent_location[1]]
        left = self.board[self.agent_location[0], self.agent_location[1] - 1]
        right = self.board[self.agent_location[0], self.agent_location[1] + 1]
        x = self.agent_location[0]
        y = self.agent_location[1]
        return (x, y, up, down, right, left)
        # return self.board.flatten().tolist()

    def get_state_q(self):
        return tuple(self.agent_location.copy())

    def loop_once(self) -> int:
        super().loop_once()
        return self.over

    def onEvent(self, event) -> None:
        if event.type == core.KEYDOWN:
            if event.key == core.K_q:
                self.running = False
                self.over = True
            if event.key == core.K_SPACE:
                self.rendering = not self.rendering

    def onRender(self) -> None:
        self._draw_grid()

    def _draw_grid(self):
        self.window.fill((0, 0, 0))
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                obj = self.board[i, j]
                x = self.box_size * j + 1
                y = self.box_size * i + 1
                core.draw.rect(self.window,
                               OBJECTS[obj]['color'],
                               (x, y, self.box_size - 1, self.box_size - 1))
                if self.model is not None:
                    if obj == E:
                        for a in range(self.action_space_size):
                            val = self.model[i][j][a]
                            color_val = 255 * (abs(val) / max(REWARDS)) if val >= 0 else 255 * (-val / -min(REWARDS))
                            if color_val > 255:
                                color_val = 255
                            color_val = 100
                            color = (0, 0, color_val) if val >= 0 else (color_val, 0, 0)
                            if a == UP:
                                core.draw.polygon(self.window, color,
                                                  [[x, y], [x + self.box_size, y],
                                                   [x + self.box_size / 2, y + self.box_size / 2]])
                            elif a == DOWN:
                                core.draw.polygon(self.window, color,
                                                  [[x + self.box_size / 2, y + self.box_size / 2],
                                                   [x, y + self.box_size], [x + self.box_size, y + self.box_size]])
                            elif a == LEFT:
                                core.draw.polygon(self.window, color,
                                                  [[x + self.box_size / 2, y + self.box_size / 2],
                                                   [x, y], [x, y + self.box_size]])
                            elif a == RIGHT:
                                core.draw.polygon(self.window, color,
                                                  [[x + self.box_size / 2, y + self.box_size / 2],
                                                   [x + self.box_size, y], [x + self.box_size, y + self.box_size]])
                        core.draw.line(self.window, (100,) * 3, (x, y), (x + self.box_size, y + self.box_size))
                        core.draw.line(self.window, (100,) * 3, (x, y + self.box_size), (x + self.box_size, y))

        for i in range(1, self.board.shape[1]):
            core.draw.line(self.window,
                           (255, 255, 255),
                           (self.box_size * i, 1),
                           (self.box_size * i, self.size[1] - 1))
        for i in range(1, self.board.shape[0]):
            core.draw.line(self.window,
                           (255, 255, 255),
                           (1, self.box_size * i),
                           (self.size[0] - 1, self.box_size * i))

    def _check_action(self, action):
        self.step_count += 1
        agent_location_tmp = self.agent_location.copy()
        if action == UP:
            agent_location_tmp[0] -= 1
        elif action == RIGHT:
            agent_location_tmp[1] += 1
        elif action == DOWN:
            agent_location_tmp[0] += 1
        elif action == LEFT:
            agent_location_tmp[1] -= 1
        obj = self.board[agent_location_tmp[0], agent_location_tmp[1]]
        if obj == E:
            return MOVABLE
        if obj == G:
            return TERMINATE
        if obj == W:
            return INPLACE
        if obj == H:
            return FALLING
        return INPLACE

    def _move(self, action):
        # Clear prev position
        self.board[self.agent_location[0], self.agent_location[1]] = E
        if action == UP:
            self.agent_location[0] -= 1
        elif action == RIGHT:
            self.agent_location[1] += 1
        elif action == DOWN:
            self.agent_location[0] += 1
        elif action == LEFT:
            self.agent_location[1] -= 1
        # Set location on board
        self.board[self.agent_location[0], self.agent_location[1]] = A

    def load_env(self, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
            board = []
            for i, row in enumerate(reader):
                encoded_row = []
                for j, cell in enumerate(row):
                    if cell == 'E':
                        encoded_row.append(E)
                    elif cell == 'W':
                        encoded_row.append(W)
                    elif cell == 'H':
                        encoded_row.append(H)
                    elif cell == 'A':
                        encoded_row.append(E)
                        self.initial_agent_location = [i, j]
                    elif cell == 'G':
                        encoded_row.append(G)
                        self.initial_goal_location = [i, j]
                board.append(encoded_row)
            self.initial_board = np.array(board)

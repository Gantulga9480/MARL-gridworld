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
E = OBJECTS[0]['id']
B = OBJECTS[1]['id']
H = OBJECTS[2]['id']
A = OBJECTS[3]['id']
G = OBJECTS[4]['id']

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
    -1,   # Empty move
    -1,   # Stayed inplace
    -10,  # Fell through hole
    10    # Found goal
]

EPISODE_TERMINATING_POLICY = [
    0,  # Empty move - RESUME
    0,  # Stayed inplace - RESUME
    1,  # Fell through hole - TERMINATE
    1   # Found goal - TERMINATE
]


class GridEnv(Game):

    def __init__(self, env_file: str = None) -> None:
        super().__init__()

        self.size = (600, 600)
        self.window_flags = 0
        self.initial_agent_location = None
        self.initial_goal_location = None
        self.state = None
        self.state_tmp = []
        self.initial_board = None
        self.board = None

        self.over = 0

        if env_file:
            self.load_env(env_file)
        else:
            self.initial_board = np.array([[E, E, E, E, G],
                                           [E, E, E, E, E],
                                           [A, E, E, E, E]])
            self.initial_agent_location = [2, 0]
            self.initial_goal_location = [0, 4]
        self.set_window()

    @property
    def state_space(self):
        return self.initial_board.shape

    @property
    def action_space(self):
        return 4

    def reset(self):
        self.over = 0
        self.state = self.initial_agent_location.copy()
        self.board = copy.deepcopy(self.initial_board)
        return tuple(self.state.copy())

    def step(self, action):
        res = self._check_action(action)
        # Valid move
        if res == MOVABLE:
            self._move()
        # Agent reached goal position
        elif res == TERMINATE:
            self._move()
        # Agent fell through hole
        elif res == FALLING:
            self._move()
        # Agent hit wall
        elif res == INPLACE:
            pass
        self.over = EPISODE_TERMINATING_POLICY[res]
        reward = REWARDS[res]
        state = self.state.copy()
        return tuple(state), reward, self.over

    def loop_once(self) -> int:
        super().loop_once()
        return self.over

    def onEvent(self, event) -> None:
        if event.type == core.KEYDOWN:
            if event.key == core.K_q:
                self.running = False
                self.over = True

    def onRender(self) -> None:
        self._draw_grid()

    def _draw_grid(self):
        self.window.fill((0, 0, 0))
        box_w = self.size[0] / (self.board.shape[1])
        box_h = self.size[1] / (self.board.shape[0])
        for i in range(1, self.board.shape[1]):
            core.draw.line(self.window,
                           (255, 255, 255),
                           (box_w*i, 1),
                           (box_w*i, self.size[1]-1))
        for i in range(1, self.board.shape[0]):
            core.draw.line(self.window,
                           (255, 255, 255),
                           (1, box_h*i),
                           (self.size[0]-1, box_h*i))
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                obj = self.board[i, j]
                core.draw.rect(self.window,
                               OBJECTS[obj]['color'],
                               (box_w*j+1, box_h*i+1, box_w-1, box_h-1))

    def _check_action(self, action):
        self.state_tmp = self.state.copy()
        if action == UP:
            self.state_tmp[0] -= 1
        elif action == RIGHT:
            self.state_tmp[1] += 1
        elif action == DOWN:
            self.state_tmp[0] += 1
        elif action == LEFT:
            self.state_tmp[1] -= 1
        if 0 <= self.state_tmp[0] < self.board.shape[0] and \
                0 <= self.state_tmp[1] < self.board.shape[1]:
            obj = self.board[self.state_tmp[0], self.state_tmp[1]]
            if obj == E:
                return MOVABLE
            if obj == G:
                return TERMINATE
            if obj == B:
                return INPLACE
            if obj == H:
                return FALLING
        return INPLACE

    def _move(self):
        # Clear prev position
        self.board[self.state[0], self.state[1]] = E
        # Set state to new location
        self.state = self.state_tmp.copy()
        # Set state on board
        self.board[self.state[0], self.state[1]] = A

    def load_env(self, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
            board = []
            for i, row in enumerate(reader):
                encoded_row = []
                for j, cell in enumerate(row):
                    if cell == 'E':
                        encoded_row.append(E)
                    elif cell == 'B':
                        encoded_row.append(B)
                    elif cell == 'H':
                        encoded_row.append(H)
                    elif cell == 'A':
                        encoded_row.append(A)
                        self.initial_agent_location = [i, j]
                    elif cell == 'G':
                        encoded_row.append(G)
                        self.initial_goal_location = [i, j]
                board.append(encoded_row)
            self.initial_board = np.array(board)

from grid import GridEnv
from RL import QLearningAgent

environment = GridEnv(env_file="boards/board3.csv")
agent = QLearningAgent(environment.state_space_size, environment.action_space_size)
agent.create_model(lr=0.1, y=0.9, e_decay=0.999)
environment.model = agent.model

while environment.running:
    state = environment.reset()
    while not environment.loop_once():
        action = agent.policy(state)
        next_state, reward, done = environment.step(action)
        agent.learn(state, action, next_state, reward, done)
        state = next_state

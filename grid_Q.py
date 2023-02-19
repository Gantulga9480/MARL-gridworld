from grid import GridEnv
from RL import QLAgent

environment = GridEnv(env_file="boards/board1.csv")
agent = QLAgent(environment.state_space_size, environment.action_space_size)
agent.create_model()
environment.model = agent.model

while environment.running:
    s = environment.reset()
    while not environment.loop_once():
        a = agent.policy(s, greedy=False)
        ns, r, d = environment.step(a)
        agent.learn(s, a, ns, r, d)
        s = ns

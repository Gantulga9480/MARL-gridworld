from grid import GridEnv
from RL import QLAgent

environment = GridEnv(env_file='boards/board3.csv')
agent = QLAgent(environment.action_space, 0.1, 0.99, 0.999)
agent.create_model((*environment.state_space, environment.action_space))

environment.table = agent.model

while environment.running:
    s = environment.reset()
    while not environment.loop_once():
        a = agent.policy(s, greedy=True)
        ns, r, d = environment.step(a)
        agent.learn(s, a, r, ns, d)
        s = ns

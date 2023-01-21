from grid import GridEnv
from RL import QLAgent

environment = GridEnv(env_file='boards/board3.csv')
agent = QLAgent()
agent.create_model((*environment.state_space, environment.action_space))

environment.table = agent.model

while environment.running:
    s = environment.reset()
    while not environment.loop_once():
        a = agent.policy(s, use_e=False)
        ns, r, d = environment.step(a)
        agent.learn(s, a, r, ns, d)
        s = ns

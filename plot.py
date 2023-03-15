import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rewards-list', nargs='+', default=[])
parser.add_argument('-w', '--window-size', type=int, default=100)
args = parser.parse_args()

if not args.rewards_list:
    quit()

rewards_files = []
window = args.window_size

for file in args.rewards_list:
    with open(file, 'r') as f:
        lines = f.readlines()
        data = [float(item.strip()) for item in lines]
        d = []
        for i in range(len(data) - window):
            d.append(sum(data[i:i + window]) / window)
        rewards_files.append(d)

fig1, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(rewards_files[0], label=args.rewards_list[0], c='g', linewidth="5")
ax2.plot(rewards_files[1], label=args.rewards_list[1], c='r', linewidth="5")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.show()

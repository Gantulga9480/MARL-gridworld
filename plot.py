import matplotlib
import matplotlib.pylab as plt
import argparse

font = {'weight': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rewards-list', nargs='+', default=[])
parser.add_argument('-w', '--window-size', type=int, default=100)
args = parser.parse_args()

if not args.rewards_list:
    quit()

rewards = []
file_names = []
window = args.window_size

for file in args.rewards_list:
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            data = [float(item.strip()) for item in lines]
            d = []
            for i in range(len(data) - window):
                d.append(sum(data[i:i + window]) / window)
            rewards.append(d)
            file_names.append(file.split(".")[0])
    except IsADirectoryError:
        continue

fig1, ax1 = plt.subplots()
for i, r in enumerate(rewards):
    ax1.plot(r, label=file_names[i], linewidth="5")
ax1.legend()
ax1.set_xlabel("Cart-Pole performance")
plt.show()

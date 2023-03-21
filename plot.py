import numpy as np
import matplotlib
import matplotlib.pylab as plt
import argparse

font = {'weight': 'normal', 'size': 18}
matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument('rewards', nargs='+')
parser.add_argument('--width', type=int, default=100)
parser.add_argument("--average", action="store_true")
parser.add_argument('--save', action="store_true")
args = parser.parse_args()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if not args.average:
    rewards_list = []
    file_names = []
    for file in args.rewards:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                rewards_list.append(np.array([float(item.strip()) for item in lines]))
                file_names.append(file.split(".")[0])
        except IsADirectoryError:
            continue

    avgs = []
    for rewards in rewards_list:
        avgs.append(moving_average(rewards, args.width))

    fig1, ax1 = plt.subplots()
    for i, r in enumerate(avgs):
        ax1.plot(r, label=file_names[i], linewidth="5")
    plt.title("Training result")
    ax1.legend()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    plt.show()

elif args.average:
    rewards = []
    file_names = []
    for file in args.rewards:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                data = [float(item.strip()) for item in lines]
                rewards.append(np.array(data))
                file_names.append(file.split(".")[0])
        except IsADirectoryError:
            continue
    np_r = np.array(rewards)
    np_r = np_r.mean(axis=0)
    avg = moving_average(np_r, args.width)

    fig1, ax1 = plt.subplots()
    ax1.plot(avg, label="Average", linewidth="5")
    ax1.legend()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    plt.title("Average training curve")
    plt.show()

    if args.save:
        with open(f'{file_names[0]}_avg.txt', 'w') as f:
            f.writelines([str(item) + '\n' for item in avg])

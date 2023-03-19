import numpy as np
import matplotlib
import matplotlib.pylab as plt
import argparse

font = {'weight': 'normal',
        'size': 18}

matplotlib.rc('font', **font)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--rewards-list', nargs='+', default=[])
parser.add_argument('-a', '--average', nargs='+', default=[])
parser.add_argument('-w', '--window-size', type=int, default=100)
args = parser.parse_args()

if args.rewards_list and args.average:
    raise ValueError("Use only one of '-r' or '-a'")

if args.rewards_list:
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

elif args.average:
    rewards = []
    file_names = []
    window = args.window_size

    for file in args.average:
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

    d = []
    for i in range(len(np_r) - window):
        d.append(np.mean(np_r[i:i + window]))

    fig1, ax1 = plt.subplots()
    ax1.plot(d, label="Average", linewidth="5")
    ax1.legend()
    ax1.set_xlabel("Cart-Pole performance")
    plt.show()

    with open(f'{file_names[0]}', 'w') as f:
        f.writelines([str(item) + '\n' for item in d])

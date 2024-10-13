import numpy as np
from scipy import stats
import os
import sys
import matplotlib.pyplot as plt


def get_latency(filename: str):
    latency_list = []
    with open(filename, "r") as f:
        for line in f.readlines():
            # if line contains "loop" ...
            if "loop" in line:
                # extract latency in ms
                latency = float(line.split()[-2])
                latency_list.append(latency)
    # remove outliers using z-score
    z = np.abs(stats.zscore(latency_list))
    threshold = 3
    latency_list = [
        latency for i, latency in enumerate(latency_list) if z[i] < threshold
    ]
    # calculate mean latency
    mean_latency = np.mean(latency_list)
    return mean_latency


def plot(results_folder: str):
    settings = ["flexnn", "wo_all", "wo_mem", "wo_preload"]
    folders = [
        "vgg19_100000000",
        "vgg19_500000000",
        "resnet152_100000000",
        "resnet152_500000000",
    ]
    labels = [
        "VGG-19\n(100MB)",
        "VGG-19\n(500MB)",
        "ResNet-152\n(100MB)",
        "ResNet-152\n(500MB)",
    ]
    # latency[setting][folder]
    latency = []
    for setting in settings:
        latency.append(
            [
                get_latency(f"{results_folder}/ablation/{folder}/{setting}.txt")
                for folder in folders
            ]
        )
    # plot latency use plt bar, x-label is the folder name, y-label is the latency, plot different setting in different color
    x = np.arange(len(folders))
    width = 0.2
    fig, ax = plt.subplots()
    for i, setting in enumerate(settings):
        ax.bar(
            x + i * width,
            latency[i],
            width,
            label=setting,
        )
    ax.set_xlabel("Model & Memory")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Abaltion Study")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()

    # plt.show()
    # savefig
    fig.savefig(f"{results_folder}/figures/ablation/ablation.png")
    print(f"ablation plot saved to {results_folder}/figures/ablation/ablation.png")

    return


if __name__ == "__main__":
    # get folder name from command line
    results_folder = sys.argv[1]
    os.makedirs(f"{results_folder}/figures/ablation", exist_ok=True)
    print(f"plotting {results_folder}/ablation")
    plot(results_folder)

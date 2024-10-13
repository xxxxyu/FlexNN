import numpy as np
import pandas as pd
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
                if line.split()[-1] == "ms":
                    latency = float(line.split()[-2])
                    latency_list.append(latency)
    # remove max and min
    latency_list = sorted(latency_list)
    latency_list = latency_list[1:-1]
    # calculate mean latency
    mean_latency = np.mean(latency_list)
    return mean_latency


def get_memory(filename: str):
    df = pd.read_csv(filename)
    # find the maximum memory usages
    max_memory = df["memory(kB)"].max()
    # find the maximum memory usage in the first n seconds
    # we've set it to idle when measuring
    idle_time = 3000  # first 3 seconds
    start_time = df["timestamp(ms)"].min()
    idle_memory = df[df["timestamp(ms)"] < start_time + idle_time]["memory(kB)"].max()

    # subtract the idle memory from the max memory
    # to get the actual memory usage of dnn inference only
    return max_memory - idle_memory


def plot_one(results_folder: str, model_name: str):
    print(f"plotting {model_name}")
    # flexnn results
    flexnn_latency = []
    flexnn_memory = []
    # get filenames in {results_folder}/end2end/flexnn/
    flexnn_files = [
        f
        for f in os.listdir(f"{results_folder}/end2end/flexnn/")
        if f"{model_name}" in f
    ]
    # extract memory from filename and latency from file content
    for f in flexnn_files:
        # remove the 'M' from memory value and convert to int
        memory = int(f.split("_")[-1].split(".")[0][:-1])
        flexnn_memory.append(memory)
        flexnn_latency.append(get_latency(f"{results_folder}/end2end/flexnn/{f}"))
    # sort latency & memory by memory
    flexnn_memory, flexnn_latency = zip(*sorted(zip(flexnn_memory, flexnn_latency)))
    # print(f"flexnn_memory: {flexnn_memory}")
    # baseline results:
    baselines = ["default", "direct", "ondemand"]
    baseline_latency = []
    baseline_memory = []  # MB
    for baseline in baselines:
        baseline_latency.append(
            get_latency(
                f"{results_folder}/end2end/baselines/{model_name}_{baseline}_latency.txt"
            )
        )
        baseline_memory.append(
            get_memory(
                f"{results_folder}/end2end/baselines/{model_name}_{baseline}_mem.csv"
            )
            / 1000
        )
    # plot x-axis: memory, y-axis: latency
    # use connected scatter for flexnn
    plt.plot(flexnn_memory, flexnn_latency, label="flexnn", marker="o")
    # use single points for 3 baselines separately
    plt.plot(baseline_memory[0], baseline_latency[0], "go", label=baselines[0])
    plt.plot(baseline_memory[1], baseline_latency[1], "bo", label=baselines[1])
    plt.plot(baseline_memory[2], baseline_latency[2], "ro", label=baselines[2])

    plt.xlabel("Memory (kB)")
    plt.ylabel("Latency (ms)")
    plt.title(f"{model_name} end2end")
    plt.legend()
    plt.savefig(f"{results_folder}/figures/end2end/{model_name}.png")
    plt.close()


if __name__ == "__main__":
    results_folder = sys.argv[1]
    os.makedirs(f"{results_folder}/figures/end2end", exist_ok=True)
    models = ["vgg19", "resnet152", "mobilenetv2", "squeezenet", "vit", "gpt2"]
    # mkdir -p results/figures/end2end
    os.makedirs(f"{results_folder}/figures/end2end", exist_ok=True)
    for model in models:
        plot_one(results_folder, model)

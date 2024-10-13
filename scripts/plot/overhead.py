import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def process_one(filename: str):
    df = pd.read_csv(filename)

    # estimate idle power
    idle_duration = 1000  # first and last 1 second
    idle_power = df[
        (df["timestamp(ms)"] < df["timestamp(ms)"].min() + idle_duration)
        | (df["timestamp(ms)"] > df["timestamp(ms)"].max() - idle_duration)
    ]["power(mW)"].mean()

    # subtract idle power from power to get the actual power consumption of dnn inference only
    df["power(mW)"] = df["power(mW)"] - idle_power

    # calculate total energy using the trapezoidal rule
    total_energy = np.trapz(df["power(mW)"], df["timestamp(ms)"])

    print(f"Total energy: {total_energy} mJ")

    df["timestamp(ms)"] = df["timestamp(ms)"] - df["timestamp(ms)"].min()

    return df


def plot(results_folder: str):

    flexnn_power = process_one(f"{results_folder}/overhead/vgg19_flexnn_power.csv")
    ncnn_power = process_one(f"{results_folder}/overhead/vgg19_ncnn_power.csv")

    # plot in one figure
    plt.figure()
    plt.plot(flexnn_power["timestamp(ms)"], flexnn_power["power(mW)"], label="flexnn")
    plt.plot(ncnn_power["timestamp(ms)"], ncnn_power["power(mW)"], label="ncnn")
    plt.legend()
    plt.xlabel("Time (ms)")
    plt.ylabel("Power (mW)")
    plt.title("Power consumption")
    plt.show()

    plt.savefig(f"{results_folder}/figures/overhead/power.png")
    print(f"Power plot saved to {results_folder}/figures/overhead/power.png")


if __name__ == "__main__":
    results_folder = sys.argv[1]
    os.makedirs(f"{results_folder}/figures/overhead", exist_ok=True)
    plot(results_folder)

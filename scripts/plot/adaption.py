import pandas as pd
import sys
import os
import matplotlib.pyplot as plt


def get_latency(filename: str):
    # read time.csv: start(ms), end(ms)
    raw_df = pd.read_csv(filename)
    # for each row, if start - last row end > 100 ms, then add 2 points
    # {last end + 50, 0}, {start - 50, 0}
    # this is to remove the idle time
    for i in range(1, len(raw_df)):
        if raw_df.loc[i, "start(ms)"] - raw_df.loc[i - 1, "end(ms)"] > 100:
            raw_df = raw_df._append(
                {
                    "start(ms)": raw_df.loc[i - 1, "end(ms)"] + 50,
                    "end(ms)": raw_df.loc[i - 1, "end(ms)"] + 50,
                },
                ignore_index=True,
            )
            raw_df = raw_df._append(
                {
                    "start(ms)": raw_df.loc[i, "start(ms)"] - 50,
                    "end(ms)": raw_df.loc[i, "start(ms)"] - 50,
                },
                ignore_index=True,
            )
    # sort by start time
    raw_df = raw_df.sort_values(by="start(ms)")

    # new df for timestamp(ms), latency(ms)
    df = pd.DataFrame(columns=["timestamp(ms)", "latency(ms)"])
    # first col: start+end/2, second col: end-start
    df["timestamp(ms)"] = (raw_df["start(ms)"] + raw_df["end(ms)"]) / 2
    df["latency(ms)"] = raw_df["end(ms)"] - raw_df["start(ms)"]

    return df


def get_memory(filename: str):
    df = pd.read_csv(filename)

    # find the maximum memory usage in the first n seconds
    # we've set it to idle when measuring
    cool_down_time = 3000  # first 3 seconds
    start_time = df["timestamp(ms)"].min()
    idle_memory = df[df["timestamp(ms)"] < start_time + cool_down_time][
        "memory(kB)"
    ].max()

    # subtract the idle memory from the max memory
    # to get the actual memory usage of dnn inference only
    df["memory(kB)"] = df["memory(kB)"] - idle_memory
    return df


def plot(latency: pd.DataFrame, memory: pd.DataFrame):

    start_time = latency["timestamp(ms)"].min()
    latency["time(ms)"] = latency["timestamp(ms)"] - start_time
    memory["time(ms)"] = memory["timestamp(ms)"] - start_time
    # remove t < 0
    latency = latency[latency["time(ms)"] >= 0]
    memory = memory[memory["time(ms)"] >= 0]

    # plot latency and memory in the same figure, time starts from 0
    # use different y-axis, ms for latency, MB for memory

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Latency (ms)", color="tab:red")
    ax1.set_xlim(0, latency["time(ms)"].max())
    ax1.set_ylim(latency["latency(ms)"].max() * 0.7, latency["latency(ms)"].max())
    ax1.plot(
        latency["time(ms)"],
        latency["latency(ms)"],
        color="tab:red",
        linewidth=2,
    )
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory (MB)", color="tab:blue")
    ax2.set_ylim(0, memory["memory(kB)"].max() / 1000 * 1.2)
    ax2.plot(
        memory["time(ms)"], memory["memory(kB)"] / 1000, color="tab:blue", linewidth=3
    )
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    fig.legend(["Latency", "Memory"], loc="upper right")

    plt.title("Adaptive latency and memory")
    plt.show()
    fig.savefig(f"{results_folder}/figures/adaption/adaption.png")
    print(f"Adaption plot saved to {results_folder}/figures/adaption/adaption.png")


if __name__ == "__main__":
    # get folder name from command line
    results_folder = sys.argv[1]
    os.makedirs(f"{results_folder}/figures/adaption", exist_ok=True)
    latency = get_latency(f"{results_folder}/adaption/time.csv")
    memory = get_memory(f"{results_folder}/adaption/mem.csv")
    plot(latency, memory)

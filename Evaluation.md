# FlexNN Evaluation Guide

This is the artifact evaluation guide for MobiCom 2024 paper *"FlexNN: Efficient and Adaptive DNN Inference on Memory-Constrained Edge Devices"*.

In this paper, we proposed FlexNN, an efficient and adaptive memory management framework for DNN inference on memory-constrained devices. FlexNN uses a slicing-loading-computing joint planning approach, to achieve optimal memory utilization and minimal memory management overhead.

We implemented FlexNN atop [NCNN](https://github.com/Tencent/ncnn), and conducted comprehensive evaluations with common model architectures on various devices. This documentation describes the complete workflow of FlexNN, and provides detailed instructions to build, deploy, and evaluate FlexNN on your device.

## Preparation

The complete evaluation will cost about 2~3 hours.

### Devices

To conduct the complete evaluation, it is recommended that you have:

- *(Optional since we provide alternative ways other than building from source)* a **host machine** (e.g., a PC) to build FlexNN and process raw results, meeting the following requirements.
  - Hardware:
    - x86-64 CPUs.
    - Available RAM >= 2GB.
    - Available Disk >= 4GB.
  - OS: Ubuntu
  - Libs: git, cmake, g++, protobuf, OpenMP.
  - Tools: ADB (for Android target), or SSH and SFTP (for Linux target).
- A **target device** (e.g., a smartphone) to deploy and evaluation FlexNN, meeting the following requirements:
  - Hardware:
    - ARMv8 CPUs (>= 2 cores).
    - Available RAM >= 2GB.
    - Available Storage >= 3GB.
  - OS: Ubuntu/Android with root access.

Note that FlexNN should also support other hardware and software configurations, but the provided workflows and scripts are verified only with the above configuration.

### Necessary Files

All necessary files to evaluate FlexNN have been uploaded to a Google Drive folder. [[Share Link]](https://drive.google.com/drive/folders/1msaJ7KZ4DGfcgn9bYSMlLUombRuaQZGH?usp=sharing) Files include:

- **models.zip**: DNN models required during evaluation. About 2.5GB in total after unzipping.
- *(Optional)* **prebuilt.zip**: Pre-built binaries of FlexNN.
- *(Optional)* **docker_flexnn.tar**: Docker image for building FlexNN and processing results. About 1GB.

## Overview

### FlexNN Workflow

With a given memory budget and DNN model, FlexNN flexibly slices the model and plans memory allocations and the execution order. The complete workflow contains the following steps (details in $\S 3$ in the paper):

- Offline Planning
  - Layer Slicing - implemented in `flexnnslice`.
    - Input: memory, model.
    - Output: sliced model.
  - Memory Profiling - implemented in `flexnnprofile`.
    - Input: sliced model.
    - Output: profiles.
  - Memory Planning - implemented in `flexnnschedule`.
    - Input: profiles.
    - Output: plans (dependencies and allocations).
- Online Execution
  - Model Inference - implemented in `benchflexnn`.
    - Input: memory, model, plans.
    - Output: inference results.

Currently, we only test the system performance.

### Evaluation Steps

The evaluation of FlexNN involves the following steps:

- [Building](#building): we provide 3 options for you to build FlexNN
  - Pre-built binaries.
  - Build with the Docker image.
  - Build with your own environment.
- [Deploying](#deployment): we provide 2 options to push files to different target OS
  - ADB for Android.
  - SSH & SFTP for Linux.
- [Conducting Experiments](#evaluation): there are 4 different groups of experiments, each with a kick-and-run script.
  - End-to-end comparison.
  - System overhead measurement.
  - Ablation study.
  - Adaptive demo.
- [Processing Raw Results](#post-processing): we provide Python scripts for data processing and visualization.

Note that the building and post-processing processes are done on a **host machine**, while the experiments are conducted on the target **mobile/edge device**.

## Building

Before continuing, ensure that you have **downloaded the [necessary files](#necessary-files).**

### Pre-built Binaries

The pre-built binaries are provided together with their building folders. After downloading and unzipping them, directly **copy the folders to the root directory** of FlexNN. For example:

```bash
# Android build
cp -r prebuilt/build-android-aarch <flexnn-root-dir>
# Linux build
cp -r prebuilt/build-aarch64-linux-gnu <flexnn-root-dir> 
```

This is to ensure that our scripts to push files to the target device can work as expected. After this, **go ahead to [deploying](#deployment).**

### Build with Docker

We provide a Ubuntu 20.04 Docker image with all dependencies installed. **Note that the image can only run on x86_64 machines.** If you use others including Mac with Apple silicon, consider using the pre-built binaries.

After [installing Docker](https://docs.docker.com/engine/install/) and downloading the provided Docker image, you can [load the provided image](https://docs.docker.com/reference/cli/docker/image/load/), and then start a container that mounts the FlexNN root directory. For example:

```bash
docker run -it -v <flexnn-root-dir>:/FlexNN <image_id>
cd /FlexNN
```

This will bring you into an interactive shell. Then, run the scripts to build for different platforms, according to your target device.

```bash
# In <flexnn-root-dir>
chmod -R 0777 ./scripts/host/

# Build for Android with NDK
./scripts/host/build-android-aarch64.sh

# Build for others with GNU toolchain 
./scripts/host/build-aarch64-linux-gnu.sh
```

In the end, you should be able to find the binaries in `<build-target>/install/bin`.

### Build from Scratch

If both of the above approaches failed, or if you are interested in further customizing the building process, you can also build FlexNN from scratch. Nevertheless, we recommend building with either native Linux or a Linux VM (including WSL).

FlexNN depends on : git, cmake, g++, protobuf, OpenMP.

On Ubuntu 20.04+, run:

```bash
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libomp-dev
```

Then follow the steps in [Build with Docker](#build-with-docker).

To further customize the building process for different platforms, **which might require considerable efforts to deal with potential bugs**, please refer to [NCNN's how-to-build guide](https://github.com/Tencent/ncnn/wiki/how-to-build) since FlexNN basically shares the same dependencies and building processes. The main difference is that we only implement for ARMv8-A (aarch64) CPUs now, so Vulkan building options are always turned off.

## Deployment

Deploying FlexNN to the device involves pushing all the required binaries, models, and scripts to the device, and setting up the workspace on the device side. Before continuing with this step, **ensure that you have downloaded and unzipped the model files**, and put the `models` folder in `<flexnn-root-dir>`.

The total size of the files is around 2.5 GB. The required time may vary from seconds to minutes, depending on how you connect to the device (e.g., USB, Wi-Fi, etc). We've provided scripts to push all the necessary files to the target device.

### Android Script

We recommend using [Android Debug Bridge (ADB)](https://developer.android.com/tools/adb) for Android devices. After installing ADB and successfully connecting to your device (either with a USB link or Wi-Fi, use `adb devices` to confirm the connection), run:

```bash
chmod -R 0777 ./scripts/host/
./scripts/host/adb-push-all.sh
```

Note that the Docker container will need access to your USB or WiFi interface to connect to the Android device.

### Linux Script

We recommend using SSH and SFTP for Linux devices. **Ensure that you have set up SSH key pair authentication to the device**, because the script uses an SFTP batch file, which is not supported by password authentication. Refer to [What is SSH Public Key Authentication?](https://www.ssh.com/academy/ssh/public-key-authentication#setting-up-public-key-authentication-for-ssh)

In `scripts/host/sftp-push-all.sh`, modify `device_tmp_dir`, `device_root_dir`, `username`, and `hostname` accordingly. Then run:

```bash
chmod -R 0777 ./scripts/host/
./scripts/host/sftp-push-all.sh
```

### Manual

In case the provided scripts don't work as expected, you can also push the necessary files manually by any possible means. Just make sure that your working directory is organized like:

```bash
- bin
  - ... (binaries)
- models
  - ncnn
    - ... (ncnn models)
  - flexnn
    - ... (flexnn models)
- profiles (empty folder)
- schedules (empty folder)
- ... (scripts)
```

Otherwise the scripts might not run the experiments normally.

## Execution

In this part, we will introduce the usage of each individual binary as mentioned in [FlexNN Workflow](#flexnn-workflow), and provide an example to automatically run the workflow. **If you are looking for steps to run the experiments, head directly to the [Evaluation](#evaluation) part.**

Typically, to slice, plan, and inference under a given memory budget, the user should run `flexnnslice`, `flexnnprofile`, `flexnnschedule`, and `benchflexnn` sequentially. In most cases, for a new memory budget, only `flexnnschedule` and `benchflexnn` need to be executed again (as is shown in `flexnndemo`), unless the earlier sliced model doesn't fit into the new memory budget.

Besides, we provide scripts (e.g., `run-flexnn.sh`) that automate the complete workflow with a given model name and memory budget ([usage here](#scripts-usage)). Since executing the binaries individually will involve a lot of intermediate files and annoying arguments to type. **We strongly recommend that you run with the script.**

### Binaries Usage

Usage of `flexnnslice`:

```bash
usage: ./bin/flexnnslice <inparam> <inbin> <outparam> <outbin> <flag> [<conv_sz> <fc_sz>]
```

Usage of `flexnnprofile`:

```bash
Usage: ./bin/flexnnprofile <model_prefix> [<key=value>...]
  model_prefix: the model path w/o .param or .bin postfix
  memory_profile_path=model_prefix.memprof
  time_profile_path=model_prefix.timeprof
  num_threads=1
  inputshape=[1,3,224,224]
  vocab_path=
Example: ./bin/flexnnprofile ~/models/flexnn/vgg19.flexnn loop_count=4 warmup_loop_count=0
```

Usage of `flexnnschedule`:

```bash
Usage: ./bin/flexnnschedule <memory_profile_path> <time_profile_path> <malloc_plan_path> <layer_dependency_path> <memory_budget> [<skip count> <memory_layout_path>]
```

Usage of `benchflexnn`:

```bash
Usage: ./bin/benchflexnn <model_prefix> [<key=value>...]
  model_prefix: the model path w/o .param or .bin postfix
  loop_count=8
  warmup_loop_count=4
  cooling_down_duration=0 (s)
  num_threads=1
  computing_powersave=2
  loading_powersave=3
  input_shape=[1,3,224,224]
  cmp_model_prefix=
  config=ncnn_parallel
  malloc_plan_path=
  layer_dependency_path=
  time_profile_path=
  memory_budget=-1
  vocab_path=
Example: ./bin/benchflexnn ~/models/flexnn/vgg19.flexnn loop_count=4 warmup_loop_count=0
```

Usage of `flexnndemo`:

```bash
Usage: ./bin/flexnndemo <ncnn_param> <ncnn_bin> <flexnn_param> <flexnn_bin> <result_path> [<idle_duration>]
```

### Scripts Usage

We automate the complete workflow through `run-flexnn.sh`. For example, If you want to inference VGG-19 under a 100 MB memory limit, run:

```bash
./run-flexnn.sh vgg19 100000000 100000000 20000000 2 [1,3,224,224] 3 0
```

In the above command:

- `vgg19` is the model name, which corresponds to the model file's name. Others: `resnet152`, `mobilenetv2`, etc.
- The first `100000000` is the total memory budget in Bytes.
- The second `100000000` is the maximum conv layer size after slicing which is with the memory budget.
- `20000000` is the maximum fc layer size after slicing which is also within the memory budget. We use a relatively small value to ensure the loading is overlapped with computing.
- `2` is the number of computing threads, which is no larger than the number of big cores.
- `[1,3,224,224]` is the input shape during evaluation, which is fixed for each model. Use `[1,3,384,384]` for vit only.
- `3` is the loading powersave mode which indicates that we prefer using the middle core for loading (1 for small, 2 for big).
- `0` is the log level (optional to use `1`, `2`).

You can modify arguments in this command to try other configurations.

## Evaluation

### Experimental Setup

We hereby provide some implementation-related experimental details. You can refer to the overall settings in $\S 5.1$ of the paper, or directly [conduct the experiments](#experiment-workflow).

#### Evaluated Models

We use the following models during evaluation. Since our approach doesn't change the model's output, and the evaluation is focused on the system performance, we use pre-trained or random weights, and random inputs for all models.

| Full Name          | Alias in Files |
| ------------------ | -------------- |
| VGG-19             | vgg19          |
| ResNet-152         | resnet152      |
| MobileNetV2        | mobilenetv2    |
| SqueezeNet         | squeezenet     |
| Vision Transformer | vit            |
| GPT-2              | gpt2           |

#### Baselines

All the baselines are based on NCNN:

- **"NCNN-Default"**: NCNN with default settings.
- **"NCNN-Direct"**: NCNN without memory-consuming optimizations.
- **"On-Demand"**: on-demand layer streaming implemented atop NCNN.

#### Key Metrics

- **Inference latency**. Measured in the program itself. Unless specified otherwise, we run 8 loops of inference with 4 loops of warm-up, and calculate the average latency of the 8 loops.
- **Memory Usage**. Measured through the `pmap` tool. Note that the memory usage of FlexNN and NCNN themselves (i.e., the idle memory usage) is ignored in the final results.
- **Energy Comsumption**. Indirectly calculated by the real-time battery voltage and current, which are obtained through the Linux sysfs.

#### Number of cores

Regarding the ARM big.LITTLE technology, we use all the big cores for computing and one little core for loading (or middle core, regarding the device specification) when running inference with FlexNN, and we use only the same number of big cores in baselines.

We don't add a little/middle core for computing in baselines because results have shown that the little/middle core will become the bottleneck and further increase inference latency in baselines.

When running on devices that don't have a little/middle core (e.g., the Raspberry Pi), we use a big core for loading instead.

### Experiment Workflow

Before running the experiments, double-check that:

- You have **root access** to the deivce.
  - Linux: `sudo -i`
  - Android: `su`
- You have given the scripts execution permission.
  - `chmod 0777 *.sh`
- You have turned off any power-saving settings.
  - See settings if you use a smartphone.
- For energy evaluation (wireless devices only, smartphones for example):
  - The device is unplugged (use ADB WiFi connection).
  - The screen is turned off.
  - There are no other background processes.

Below are instructions to run the experiments step-by-step, where `<n>` is the number of big cores for computing. You can keep other arguments as is. Note that the 4-second idle duration is for collecting idle memory usage to calculate the actual memory used for DNN inference only.

#### End-to-End

Estimated time: 0.5~1 hour. There is a sleep duration between sets of experiments to avoid overheating.

```bash
./eval-end-to-end.sh <n> 3 10 4

# Usage: exec <num_threads> [<loading_powersave> <sleep_duration> <idle_duration>]
```

#### System Overhead

Estimated time: < 10 minutes.

Important: The script uses sysfs interface to get the real-time voltage and current of the battery, and then calculates the power. However, on different devices, the battery path (`/sys/class/power_supply/battery` on Google Pixel 6 Pro), the units (uV and uA on Google Pixel 6 Pro), and the positive/negative current's meaning (current > 0 means charging and current < 0 means discharging) might be different. Please check for your device and modify `eval-energy.sh` accordingly.

```bash
./eval-overhead.sh <n> 3

# Usage: exec <num_threads> [<loading_powersave>]
```

#### Ablation Study

Estimated time: < 10 minutes.

```bash
./eval-ablation.sh <n> 3

# Usage: exec <num_threads> [<loading_powersave>]
```

#### Adaptive Demo

Estimated time: < 10 minutes.

```bash
./eval-adaption.sh <n> 3 4

# Usage: exec <num_threads> [<loading_powersave> <idle_duration>]
```

#### Retrieving Raw Results

After conducting all the experiments, you will see all the raw results under the `results` folder in your device-side workspace. We provide scripts to retrieve these raw results.

For Android devices with ADB connection, run:

```bash
chmod -R 0777 ./scripts/host/
./scripts/host/adb-pull-all.sh
```

For other Linux devices with SFTP connection, run:

```bash
chmod -R 0777 ./scripts/host/
./scripts/host/sftp-pull-all.sh
```

They will be pulled to `flexnn-root-dir/{results_datetime}` on your host machine.

### Post-Processing

The figures in the paper are manually made with Excel, which requires considerable effort to reproduce. As an alternative, we provide Python scripts to process and visualize these results to demonstrate their reproducibility.

#### Steps

Requirements to run the Python scripts (pre-installed in the provided Docker image):

```bash
# if you don't have python installed
# apt-get update
# apt-get install -y python3-pip
pip3 install matplotlib numpy pandas scipy
```

In `<flexnn-root-dir>`, run:

```bash
# <results_folder> is the retrieved results (root) folder
python ./scripts/plot/end2end.py <results_folder>
python ./scripts/plot/overhead.py <results_folder>
python ./scripts/plot/ablation.py <results_folder>
python ./scripts/plot/adaption.py <results_folder>
```

After executing the scripts, the figures will be stored in `<results_folder>/figures`. Besides, you can directly read the time and storage overhead in `<results_folder>/overhead/time.csv` and `<results_folder>/overhead/size.csv`.

#### Expected Results

Generally, you are expected to see figures similar to the paper. Regarding each experiment:

- End-to-end: you should see FlexNN achieving better latency-memory tradeoffs. Specifically, FlexNN's curve is on the down left of the x-y plane.
- Overhead: you should see a time overhead of around 1 second, depending on the device's performance, and there should be about a 5~10% increase in energy consumption of FlexNN.
- Ablation: you should see FlexNN outperforming the others, with the lowest latency.
- Adaptive Demo: you should see FlexNN achieving lower latency with a higher memory budget, and the actual memory usage is strictly limited within the given memory budget.

Acceptable variation of the results:

- Latency: there might be performance fluctuations within a 5~10% range, depending on the target device. Ensuring that there are no background apps will to some extent mitigate this issue.
- Memory: there might be a 5% varying range for measured peak memory of baselines, depending on the real-time sampling rate. Note that FlexNN's memory is pre-allocated and won't change dynamically, so the measured results are very accurate.
- Energy: the energy result is highly dependent on the device, but you should at least see FlexNN having higher energy consumption than NCNN.

## Limitations

There are some known issues and limitations with the current FlexNN implementation:

- Performance fluctuations. According to the evaluation results, the latency of FlexNN tends to have a larger variance than that of NCNN. This indicates that despite the latency saving, the computing-loading overlapping scheme might be less robust against a dynamic runtime environment.
- Limited operators. FlexNN is designed to support a wide range of DNN models and operators. However, the current implementation is mainly implemented and optimized for "classic" CNNs including VGGNets, ResNets, MobileNets and SqueezeNets. The user would need to mannually add the operators and slicing strategy to support new models.
- GPT-2 slicing bug. The aarch64-built `flexnnslice` might encounter a segmentation fault when slicing the GPT-2 model (and GPT-2 only). We've provided models sliced with the x86_64-built `flexnnslice`, which works fine, to avoid this issue. The reason is not clear yet since `flexnnslice` doesn't involve platform-specific codes
(e.g., #ifdef aarch64).

GitHub issues and emails are welcome if you find any other issues.

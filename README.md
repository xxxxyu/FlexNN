# FlexNN

This repository contains the code and artifact evaluation guide for *"FlexNN: Efficient and Adaptive DNN Inference on Memory-Constrained Edge Devices"* (**ACM MobiCom 2024**). [[paper]](https://dl.acm.org/doi/abs/10.1145/3636534.3649391) [[slides]](assets/FlexNN-MobiCom.pdf)

![paper-title](<assets/paper-title.png>)

The artifact has been awarded all the 4 badges (*Artifacts Available*, *Artifacts Evaluated - Functional*, *Artifacts Evaluated - Reusable*, *Results Replicated*) in the MobiCom 2024 Artifact Evaluation. Here are the main evaluation results:

![end-to-end](<assets/end-to-end.png>)

Please refer to [Evaluation.md](Evaluation.md) for detailed instructions to evaluate FlexNN and reproduce the results.

## Introduction

FlexNN is an efficient and adaptive memory management framework for DNN inference on memory-constrained devices.

For a given memory budget and DNN model, it uses a *slicing-loading-computing joint planning* approach, to achieve optimal memory utilization with minimal memory management overhead.

## Design

![overview](<assets/overview.png>)

FlexNN adopts a 2-stages design: the offline planning stage performs *slicing-loading-computing
joint planning* according to the memory budget and the given model, and the online execution stage conducts model inference based on the offline-generated plans. It is efficient and adaptive with the following designs:

- **Bottleneck-aware layer slicing** ($\S 3.2$) that significantly reduces the peak memory usage through fine-grained partitioning on "bottleneck" layers.
- **Preload-aware memory planning** ($\S 3.3$) that reduces memory fragments and I/O waiting time by considering the weights preloading process at the offline planning stage.
- **Online execution designs** (dependency-based synchronization and type-based static allocation, in $\S 3.4$) that fill the gaps between the planning results and the actual execution.

For more details, please refer to our [paper](https://dl.acm.org/doi/pdf/10.1145/3636534.3649391).

## Implementation

FlexNN is built atop [NCNN](https://github.com/Tencent/ncnn), which is a high-performance Neural Network inference framework optimized for the mobile platform. It is implemented and best-optimized for floating-point CNN inference on ARMv8 CPUs.

With a given memory budget and DNN model, FlexNN flexibly slices the model and plans memory allocations and the execution order. The complete steps and corresponding compiled binaries are listed as follows:

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

For more details, please refer to our [paper](https://dl.acm.org/doi/pdf/10.1145/3636534.3649391) and [Evaluation.md](Evaluation.md).

## Evaluation

Please refer to [Evaluation.md](Evaluation.md) for detailed instructions to deploy and evaluate FlexNN.

The current FlexNN implementation is mainly for research purpose and has a few limitations. Please also refer to [Evaluation.md](Evaluation.md) for details.

## Citation

If you find FlexNN useful for your research, please cite our [paper](https://dl.acm.org/doi/pdf/10.1145/3636534.3649391).

```bibtex
@inproceedings{li2024flexnn,
  title={FlexNN: Efficient and Adaptive DNN Inference on Memory-Constrained Edge Devices},
  author={Li, Xiangyu and Li, Yuanchun and Li, Yuanzhe and Cao, Ting and Liu, Yunxin},
  booktitle={Proceedings of the 30th Annual International Conference on Mobile Computing and Networking},
  pages={709--723},
  year={2024}
}
```

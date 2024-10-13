for args in "$@"; do
    case $args in
        "model_name")
            echo "- model_name: name of the model that matches the file name. e.g., vgg19, resnet152, mobilenetv2, squeezenet, vit, gpt2, etc."
            ;;
        "memory_budget")
            echo "- memory_budget: memory budget for the model in Bytes."
            ;;
        "conv_sz")
            echo "- conv_sz: maximum memory size of the convolutional layer in Bytes, usually set to the same as memory_budget."
            ;;
        "fc_sz")
            echo "- fc_sz: maximum memory size of the fully connected layer in Bytes, usually set to around 1/5 of the memory_budget."
            ;;
        "num_threads")
            echo "- num_threads: number of computing threads. computing is fixed on big cores, so don't exceed big cores number."
            ;;
        "input_shape")
            echo "- input_shape: input shape of the model. use [1,3,224,224] for CNNs, and [1,3,384,384] for vit."
            ;;
        # "log_level")
        #     echo "  log_level: 0 for no log, 1 for log."
        #     ;;
        "loading_powersave")
            echo "- loading_powersave: type of core to run the loading thread. 1 for little, 2 for big, 3 for middle (if have). use 1/3 in most cases, according to your platform."
            ;;
        "sleep_duration")
            echo "- sleep_duration: sleep duration between evaluation sets."
            ;;
        "cooling_down_duration")
            echo "- cooling_down_duration: duration to cool down the device between evaluations. for memory evaluation only to collect idle memory"
            ;;
        "skip_slicing")
            echo "- skip_slicing: 1 to skip slicing, 0 to slice. default to 0. use 1 if you have prepared the sliced model."
            ;;
    esac
done
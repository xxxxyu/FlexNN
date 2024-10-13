# help
if [ $# -lt 1 ]; then
    echo "Usage: exec <num_threads> [<loading_powersave> <sleep_duration> <cooling_down_duration>]"
    chmod 0777 print-help.sh
    ./print-help.sh num_threads loading_powersave sleep_duration cooling_down_duration
    exit -1
fi

echo "eval-end-to-end.sh"

# args
n=$1 # number of computing threads
loading_powersave=3
if [ $# -ge 2 ]; then
    loading_powersave=$2
fi
sleep_duration=10 # sleep time between two runs
if [ $# -ge 3 ]; then
    sleep_duration=$3
fi
cooling_down_duration=0 # cool down time before inference, used for collecting idle memory
if [ $# -ge 4 ]; then
    cooling_down_duration=$4
fi

chmod 0777 run-flexnn.sh
chmod 0777 run-gpt2.sh
chmod 0777 measure-mem.sh

# results folder
mkdir -p results/end2end/flexnn
mkdir -p results/end2end/baselines

echo "Start to evaluate FlexNN's end-to-end performance."
global_start_time=$(date +%s%3N)

### ---------------<FlexNN's memory budget and latency>--------------- ###
echo "Start to evaluate FlexNN's memory budget and latency."
# VGG-19
./run-flexnn.sh vgg19 60000000 35000000 10000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_60M.txt 2>&1
./run-flexnn.sh vgg19 70000000 35000000 10000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_70M.txt 2>&1
./run-flexnn.sh vgg19 80000000 35000000 10000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_80M.txt 2>&1
./run-flexnn.sh vgg19 90000000 35000000 10000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_90M.txt 2>&1
./run-flexnn.sh vgg19 100000000 100000000 20000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_100M.txt 2>&1
./run-flexnn.sh vgg19 200000000 200000000 40000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_200M.txt 2>&1
./run-flexnn.sh vgg19 300000000 200000000 60000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_300M.txt 2>&1
./run-flexnn.sh vgg19 400000000 200000000 80000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_400M.txt 2>&1
./run-flexnn.sh vgg19 600000000 200000000 100000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_600M.txt 2>&1
./run-flexnn.sh vgg19 850000000 200000000 150000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_850M.txt 2>&1
./run-flexnn.sh vgg19 1000000000 200000000 200000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/vgg19_1000M.txt 2>&1

sleep $sleep_duration

# ResNet-152
./run-flexnn.sh resnet152 35000000 35000000 7000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_35M.txt 2>&1
./run-flexnn.sh resnet152 50000000 50000000 10000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_50M.txt 2>&1
./run-flexnn.sh resnet152 100000000 100000000 20000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_100M.txt 2>&1
./run-flexnn.sh resnet152 200000000 200000000 40000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_200M.txt 2>&1
./run-flexnn.sh resnet152 300000000 200000000 60000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_300M.txt 2>&1
./run-flexnn.sh resnet152 400000000 200000000 80000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_400M.txt 2>&1
./run-flexnn.sh resnet152 500000000 200000000 100000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_500M.txt 2>&1
./run-flexnn.sh resnet152 600000000 200000000 100000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/resnet152_600M.txt 2>&1

sleep $sleep_duration

# SqueezeNet
./run-flexnn.sh squeezenet 10000000 10000000 2000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/squeezenet_10M.txt 2>&1
./run-flexnn.sh squeezenet 15000000 15000000 3000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/squeezenet_15M.txt 2>&1
./run-flexnn.sh squeezenet 20000000 20000000 4000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/squeezenet_20M.txt 2>&1

sleep $sleep_duration

# MobileNet-V2
./run-flexnn.sh mobilenetv2 12000000 12000000 2000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/mobilenetv2_12M.txt 2>&1
./run-flexnn.sh mobilenetv2 15000000 15000000 3000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/mobilenetv2_15M.txt 2>&1
./run-flexnn.sh mobilenetv2 20000000 20000000 4000000 $n [1,3,224,224] $loading_powersave 0 > results/end2end/flexnn/mobilenetv2_20M.txt 2>&1

sleep $sleep_duration

# ViT
./run-flexnn.sh vit 200000000 200000000 40000000 $n [1,3,384,384] $loading_powersave 0 > results/end2end/flexnn/vit_200M.txt 2>&1
./run-flexnn.sh vit 300000000 300000000 60000000 $n [1,3,384,384] $loading_powersave 0 > results/end2end/flexnn/vit_300M.txt 2>&1
./run-flexnn.sh vit 500000000 500000000 100000000 $n [1,3,384,384] $loading_powersave 0 > results/end2end/flexnn/vit_500M.txt 2>&1

sleep $sleep_duration

# GPT-2
# Note that there is a slicing bug in GPT-2, see README for details
./run-gpt2.sh gpt2_370 370000000 370000000 70000000 $n $loading_powersave 0 1 > results/end2end/flexnn/gpt2_370M.txt 2>&1
./run-gpt2.sh gpt2_400 400000000 400000000 80000000 $n $loading_powersave 0 1 > results/end2end/flexnn/gpt2_400M.txt 2>&1
./run-gpt2.sh gpt2_500 500000000 500000000 100000000 $n $loading_powersave 0 1 > results/end2end/flexnn/gpt2_500M.txt 2>&1

sleep $sleep_duration

### ---------------<baselines' latency>--------------- ###
echo "Start to evaluate baselines' latency."
# VGG-19
./bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_default > results/end2end/baselines/vgg19_default_latency.txt 2>&1
./bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_direct_conv > results/end2end/baselines/vgg19_direct_latency.txt 2>&1
./bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_ondemand > results/end2end/baselines/vgg19_ondemand_latency.txt 2>&1

sleep $sleep_duration

# ResNet-152
./bin/benchflexnn models/ncnn/resnet152.ncnn num_threads=$n config=ncnn_default > results/end2end/baselines/resnet152_default_latency.txt 2>&1
./bin/benchflexnn models/ncnn/resnet152.ncnn num_threads=$n config=ncnn_direct_conv > results/end2end/baselines/resnet152_direct_latency.txt 2>&1
./bin/benchflexnn models/ncnn/resnet152.ncnn num_threads=$n config=ncnn_ondemand > results/end2end/baselines/resnet152_ondemand_latency.txt 2>&1

sleep $sleep_duration

# SqueezeNet
./bin/benchflexnn models/ncnn/squeezenet.ncnn num_threads=$n config=ncnn_default > results/end2end/baselines/squeezenet_default_latency.txt 2>&1
./bin/benchflexnn models/ncnn/squeezenet.ncnn num_threads=$n config=ncnn_direct_conv > results/end2end/baselines/squeezenet_direct_latency.txt 2>&1
./bin/benchflexnn models/ncnn/squeezenet.ncnn num_threads=$n config=ncnn_ondemand > results/end2end/baselines/squeezenet_ondemand_latency.txt 2>&1

sleep $sleep_duration

# MobileNet-V2
./bin/benchflexnn models/ncnn/mobilenetv2.ncnn num_threads=$n config=ncnn_default > results/end2end/baselines/mobilenetv2_default_latency.txt 2>&1
./bin/benchflexnn models/ncnn/mobilenetv2.ncnn num_threads=$n config=ncnn_direct_conv > results/end2end/baselines/mobilenetv2_direct_latency.txt 2>&1
./bin/benchflexnn models/ncnn/mobilenetv2.ncnn num_threads=$n config=ncnn_ondemand > results/end2end/baselines/mobilenetv2_ondemand_latency.txt 2>&1

sleep $sleep_duration

# ViT
./bin/benchflexnn models/ncnn/vit.ncnn num_threads=$n input_shape=[1,3,384,384] config=ncnn_default > results/end2end/baselines/vit_default_latency.txt 2>&1
./bin/benchflexnn models/ncnn/vit.ncnn num_threads=$n input_shape=[1,3,384,384] config=ncnn_direct_conv > results/end2end/baselines/vit_direct_latency.txt 2>&1
./bin/benchflexnn models/ncnn/vit.ncnn num_threads=$n input_shape=[1,3,384,384] config=ncnn_ondemand > results/end2end/baselines/vit_ondemand_latency.txt 2>&1

sleep $sleep_duration

# GPT-2
./bin/benchflexnn models/ncnn/gpt2.ncnn num_threads=$n vocab_path=models/ncnn/gpt2-vocab.txt config=ncnn_default > results/end2end/baselines/gpt2_default_latency.txt 2>&1
./bin/benchflexnn models/ncnn/gpt2.ncnn num_threads=$n vocab_path=models/ncnn/gpt2-vocab.txt config=ncnn_direct_conv > results/end2end/baselines/gpt2_direct_latency.txt 2>&1
./bin/benchflexnn models/ncnn/gpt2.ncnn num_threads=$n vocab_path=models/ncnn/gpt2-vocab.txt config=ncnn_ondemand > results/end2end/baselines/gpt2_ondemand_latency.txt 2>&1

sleep $sleep_duration

### ---------------<baselines' memory usage>--------------- ###
echo "Start to evaluate baselines' memory usage."
# VGG-19
./measure-mem.sh "bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_default cooling_down_duration=$cooling_down_duration" results/end2end/baselines/vgg19_default_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_direct_conv cooling_down_duration=$cooling_down_duration" results/end2end/baselines/vgg19_direct_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_ondemand cooling_down_duration=$cooling_down_duration" results/end2end/baselines/vgg19_ondemand_mem.csv

sleep $sleep_duration

# ResNet-152
./measure-mem.sh "bin/benchflexnn models/ncnn/resnet152.ncnn num_threads=$n config=ncnn_default cooling_down_duration=$cooling_down_duration" results/end2end/baselines/resnet152_default_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/resnet152.ncnn num_threads=$n config=ncnn_direct_conv cooling_down_duration=$cooling_down_duration" results/end2end/baselines/resnet152_direct_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/resnet152.ncnn num_threads=$n config=ncnn_ondemand cooling_down_duration=$cooling_down_duration" results/end2end/baselines/resnet152_ondemand_mem.csv

sleep $sleep_duration

# SqueezeNet
./measure-mem.sh "bin/benchflexnn models/ncnn/squeezenet.ncnn num_threads=$n config=ncnn_default cooling_down_duration=$cooling_down_duration" results/end2end/baselines/squeezenet_default_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/squeezenet.ncnn num_threads=$n config=ncnn_direct_conv cooling_down_duration=$cooling_down_duration" results/end2end/baselines/squeezenet_direct_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/squeezenet.ncnn num_threads=$n config=ncnn_ondemand cooling_down_duration=$cooling_down_duration" results/end2end/baselines/squeezenet_ondemand_mem.csv

sleep $sleep_duration

# MobileNet-V2
./measure-mem.sh "bin/benchflexnn models/ncnn/mobilenetv2.ncnn num_threads=$n config=ncnn_default cooling_down_duration=$cooling_down_duration" results/end2end/baselines/mobilenetv2_default_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/mobilenetv2.ncnn num_threads=$n config=ncnn_direct_conv cooling_down_duration=$cooling_down_duration" results/end2end/baselines/mobilenetv2_direct_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/mobilenetv2.ncnn num_threads=$n config=ncnn_ondemand cooling_down_duration=$cooling_down_duration" results/end2end/baselines/mobilenetv2_ondemand_mem.csv

sleep $sleep_duration

# ViT
./measure-mem.sh "bin/benchflexnn models/ncnn/vit.ncnn num_threads=$n input_shape=[1,3,384,384] config=ncnn_default cooling_down_duration=$cooling_down_duration" results/end2end/baselines/vit_default_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/vit.ncnn num_threads=$n input_shape=[1,3,384,384] config=ncnn_direct_conv cooling_down_duration=$cooling_down_duration" results/end2end/baselines/vit_direct_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/vit.ncnn num_threads=$n input_shape=[1,3,384,384] config=ncnn_ondemand cooling_down_duration=$cooling_down_duration" results/end2end/baselines/vit_ondemand_mem.csv

sleep $sleep_duration

# GPT-2
./measure-mem.sh "bin/benchflexnn models/ncnn/gpt2.ncnn num_threads=$n vocab_path=models/ncnn/gpt2-vocab.txt config=ncnn_default cooling_down_duration=$cooling_down_duration" results/end2end/baselines/gpt2_default_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/gpt2.ncnn num_threads=$n vocab_path=models/ncnn/gpt2-vocab.txt config=ncnn_direct_conv cooling_down_duration=$cooling_down_duration" results/end2end/baselines/gpt2_direct_mem.csv
./measure-mem.sh "bin/benchflexnn models/ncnn/gpt2.ncnn num_threads=$n vocab_path=models/ncnn/gpt2-vocab.txt config=ncnn_ondemand cooling_down_duration=$cooling_down_duration" results/end2end/baselines/gpt2_ondemand_mem.csv

global_end_time=$(date +%s%3N)
global_duration=$(($global_end_time - $global_start_time))
echo "End-to-end performance evaluation finished. Elapsed time: $global_duration ms."
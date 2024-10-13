# help
if [ $# -lt 1 ]; then
    echo "Usage: exec <num_threads> [<loading_powersave>]"
    chmod 0777 print-help.sh
    ./print-help.sh num_threads loading_powersave
    exit -1
fi

echo "eval-ablation.sh"

# args
n=$1 # number of computing threads
loading_powersave=3
if [ $# -ge 2 ]; then
    loading_powersave=$2
fi

chmod 0777 run-one-ablation.sh

echo "Start to evaluate ablation study."
global_start_time=$(date +%s%3N)

# VGG-19 100MB
./run-one-ablation.sh vgg19 100000000 100000000 20000000 $n "[1,3,224,224]" $loading_powersave
# VGG-19 500MB
./run-one-ablation.sh vgg19 500000000 200000000 100000000 $n "[1,3,224,224]" $loading_powersave
# ResNet-152 100MB
./run-one-ablation.sh resnet152 100000000 100000000 20000000 $n "[1,3,224,224]" $loading_powersave
# ResNet-152 500MB
./run-one-ablation.sh resnet152 500000000 200000000 100000000 $n "[1,3,224,224]" $loading_powersave

global_end_time=$(date +%s%3N)
global_duration=$(($global_end_time - $global_start_time))
echo "Ablation study finished. Elapsed time: $global_duration ms."
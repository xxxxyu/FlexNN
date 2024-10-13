# help
if [ $# -lt 1 ]; then
    echo "Usage: exec <num_threads> [<loading_powersave> <idle_duration>]"
    chmod 0777 print-help.sh
    ./print-help.sh num_threads loading_powersave idle_duration
    exit -1
fi

echo "eval-adaption.sh"

# args
n=$1 # number of computing threads
loading_powersave=3
if [ $# -ge 2 ]; then
    loading_powersave=$2
fi
idle_duration=0 # cool down time before inference, used for collecting idle memory
if [ $# -ge 3 ]; then
    idle_duration=$3
fi

mkdir -p results/adaption

chmod 0777 measure-mem.sh
chmod 0777 bin/flexnndemo

echo "Start to evaluate adaptive demo."

global_start_time=$(date +%s%3N)

./measure-mem.sh "./bin/flexnndemo models/ncnn/vgg19.ncnn.param models/ncnn/vgg19.ncnn.bin models/flexnn/vgg19.flexnn.param models/flexnn/vgg19.flexnn.bin results/adaption/time.csv $idle_duration" results/adaption/mem.csv

global_end_time=$(date +%s%3N)
global_duration=$(($global_end_time - $global_start_time))
echo "Adaptive demo finished. Elapsed time: $global_duration ms."
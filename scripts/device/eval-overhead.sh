# help
if [ $# -lt 1 ]; then
    echo "Usage: exec <num_threads> [<loading_powersave>]"
    chmod 0777 print-help.sh
    ./print-help.sh num_threads loading_powersave
    exit -1
fi

echo "eval-overhead.sh"

# args
n=$1 # number of computing threads
loading_powersave=3
if [ $# -ge 2 ]; then
    loading_powersave=$2
fi

# chmod
chmod 0777 run-flexnn.sh
chmod 0777 measure-energy.sh

# results path
mkdir -p results/overhead
echo "model,storage" > results/overhead/size.csv
echo "model,slicing(ms),profiling(ms),scheduling(ms)" > results/overhead/time.csv
# > results/overhead/vgg19_flexnn_power.csv
# > results/overhead/vgg19_ncnn_power.csv

global_start_time=$(date +%s%3N)

### ---------------<storage and time overhead>--------------- ###

# run the workflow and write logs to tmp files
./run-flexnn.sh vgg19 100000000 100000000 20000000 $n [1,3,224,224] $loading_powersave 1 > "vgg19_overhead.txt" 2>&1
./run-flexnn.sh resnet152 100000000 100000000 20000000 $n [1,3,224,224] $loading_powersave 1 > "resnet152_overhead.txt" 2>&1
./run-flexnn.sh vit 300000000 300000000 60000000 $n [1,3,384,384] $loading_powersave 1 > "vit_overhead.txt" 2>&1

# get model size
du -h models/flexnn/vgg19.flexnn.bin | awk '{print "vgg19," $1}' >> results/overhead/size.csv
du -h models/flexnn/resnet152.flexnn.bin | awk '{print "resnet152," $1}' >> results/overhead/size.csv
du -h models/flexnn/vit.flexnn.bin | awk '{print "vit," $1}' >> results/overhead/size.csv
echo "Model storage results saved to results/overhead/size.csv"

# get time
for model in vgg19 resnet152 vit; do
    slicing=$(cat ${model}_overhead.txt | grep "total slicing time" | awk '{print $4}')
    profiling=$(cat ${model}_overhead.txt | grep "total profiling time" | awk '{print $4}')
    scheduling=$(cat ${model}_overhead.txt | grep "total scheduling time" | awk '{print $4}')
    echo "$model,$slicing,$profiling,$scheduling" >> results/overhead/time.csv
done
echo "Time results saved to results/overhead/time.csv"

# rm tmp files
rm vgg19_overhead.txt resnet152_overhead.txt vit_overhead.txt

### ---------------<energy overhead>--------------- ###

echo "Energy test begin. Ensure the device is unplugged."

./measure-energy.sh "./bin/benchflexnn models/flexnn/vgg19.flexnn malloc_plan_path=schedules/vgg19.flexnn.malloc layer_dependency_path=schedules/vgg19.flexnn.malloc layer_dependency_path=schedules/vgg19.flexnn.dependency memory_budget=100000000 config=flexnn_parallel num_threads=$n loading_powersave=$loading_powersave computing_powersave=2 input_shape=[1,3,224,224]" "results/overhead/vgg19_flexnn_power.csv"

./measure-energy.sh "./bin/benchflexnn models/ncnn/vgg19.ncnn num_threads=$n config=ncnn_default" "results/overhead/vgg19_ncnn_power.csv"

global_end_time=$(date +%s%3N)
global_duration=$(($global_end_time - $global_start_time))
echo "Overhead evaluation finished. Elapsed time: $global_duration ms."
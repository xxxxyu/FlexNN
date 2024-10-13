# help
if [ $# -lt 6 ]; then
    echo "Usage: exec <model_name> <memory_budget> <conv_sz> <fc_sz> <num_threads> [<input_shape> <loading_powersave>]"
    chmod 0777 print-help.sh
    ./print-help.sh model_name memory_budget conv_sz fc_sz num_threads input_shape loading_powersave
    exit -1
fi

echo "run-one-ablation.sh"

# init args
model_name=$1
memory_budget=$2
conv_sz=$3
fc_sz=$4
num_threads=$5
input_shape="[1,3,224,224]"
loading_powersave=3

if [ $# -ge 6 ]; then
    input_shape=$6
fi

if [ $# -ge 7 ]; then
    loading_powersave=$7
fi

# create folder if not exist
results_folder=results/ablation/${model_name}_${memory_budget}
mkdir -p $results_folder

# chmod
chmod 0777 bin/flexnnslice
chmod 0777 bin/flexnnprofile
chmod 0777 bin/flexnnschedule
chmod 0777 bin/benchflexnn

# commands to run flexnn binaries
cmd_slice="./bin/flexnnslice models/ncnn/$model_name.ncnn.param models/ncnn/$model_name.ncnn.bin models/flexnn/$model_name.flexnn.param models/flexnn/$model_name.flexnn.bin 0 $conv_sz $fc_sz"
cmd_prof="./bin/flexnnprofile models/flexnn/$model_name.flexnn memory_profile_path=profiles/$model_name.flexnn.memprof time_profile_path=profiles/$model_name.flexnn.timeprof input_shape=$input_shape num_threads=$num_threads"
cmd_sched="./bin/flexnnschedule profiles/$model_name.flexnn.memprof profiles/$model_name.flexnn.timeprof schedules/$model_name.flexnn.malloc schedules/$model_name.flexnn.dependency $memory_budget 1 results/$model_name.flexnn.layout"
cmd_bench="./bin/benchflexnn models/flexnn/$model_name.flexnn malloc_plan_path=schedules/$model_name.flexnn.malloc layer_dependency_path=schedules/$model_name.flexnn.dependency memory_budget=$memory_budget config=flexnn_parallel num_threads=$num_threads loading_powersave=$loading_powersave computing_powersave=2 input_shape=$input_shape"
cmd_bench_wo_mem="./bin/benchflexnn models/flexnn/$model_name.flexnn layer_dependency_path=schedules/$model_name.flexnn.dependency config=flexnn_parallel num_threads=$num_threads loading_powersave=$loading_powersave computing_powersave=2 input_shape=$input_shape"
cmd_bench_wo_preload="./bin/benchflexnn models/flexnn/$model_name.flexnn malloc_plan_path=schedules/$model_name.flexnn.malloc memory_budget=$memory_budget config=flexnn_ondemand num_threads=$num_threads loading_powersave=$loading_powersave computing_powersave=2 input_shape=$input_shape"
cmd_bench_wo_all="./bin/benchflexnn models/flexnn/$model_name.flexnn config=flexnn_ondemand num_threads=$num_threads loading_powersave=$loading_powersave computing_powersave=2 input_shape=$input_shape"

# mute offline planning logs by default
# $cmd_slice
$cmd_slice > /dev/null 2>&1
# $cmd_prof
$cmd_prof > /dev/null 2>&1
# $cmd_sched
$cmd_sched > /dev/null 2>&1
# $cmd_bench
# echo "  Model: $model_name; memory budget: $memory_budget; config: FlexNN"
$cmd_bench 2>&1 | grep "models/flexnn/$model_name.flexnn" > $results_folder/flexnn.txt
# echo "  Model: $model_name; memory budget: $memory_budget; config: w.o.mem"
$cmd_bench_wo_mem 2>&1 | grep "models/flexnn/$model_name.flexnn" > $results_folder/wo_mem.txt
# echo "  Model: $model_name; memory budget: $memory_budget; config: w.o.preload"
$cmd_bench_wo_preload 2>&1 | grep "models/flexnn/$model_name.flexnn" > $results_folder/wo_preload.txt
# echo "  Model: $model_name; memory budget: $memory_budget; config: w.o.both"
$cmd_bench_wo_all 2>&1 | grep "models/flexnn/$model_name.flexnn" > $results_folder/wo_all.txt

# help
if [ $# -lt 4 ]; then
    echo "Usage: exec <model_name> <memory_budget> <conv_sz> <fc_sz> <num_threads> [<loading_powersave> <log_level> <skip_slicing>]"
    chmod 0777 print-help.sh
    ./print-help.sh model_name memory_budget conv_sz fc_sz num_threads loading_powersave log_level skip_slicing
    exit -1
fi

echo "run-gpt.sh"

# init args
model_name=$1
memory_budget=$2
conv_sz=$3
fc_sz=$4
num_threads=$5
loading_powersave=3
log_level=0
skip_slicing=0

if [ $# -ge 6 ]; then
    loading_powersave=$6
fi

if [ $# -ge 7 ]; then
    log_level=$7
fi

if [ $# -ge 8 ]; then
    skip_slicing=$8
fi

# chmod
chmod 0777 bin/flexnnslice
chmod 0777 bin/flexnnprofile
chmod 0777 bin/flexnnschedule
chmod 0777 bin/benchflexnn

# commands to run flexnn binaries
cmd_slice="./bin/flexnnslice models/ncnn/$model_name.ncnn.param models/ncnn/$model_name.ncnn.bin models/flexnn/$model_name.flexnn.param models/flexnn/$model_name.flexnn.bin 0 $conv_sz $fc_sz"
if [ $skip_slicing -eq 1 ]; then
    cmd_slice=""
fi
cmd_prof="./bin/flexnnprofile models/flexnn/$model_name.flexnn memory_profile_path=profiles/$model_name.flexnn.memprof time_profile_path=profiles/$model_name.flexnn.timeprof input_shape=$input_shape num_threads=$num_threads vocab_path=$home_dir/models/ncnn/gpt2-vocab.txt"
cmd_sched="./bin/flexnnschedule profiles/$model_name.flexnn.memprof profiles/$model_name.flexnn.timeprof schedules/$model_name.flexnn.malloc schedules/$model_name.flexnn.dependency $memory_budget 2 results/$model_name.flexnn.layout"
cmd_bench="./bin/benchflexnn models/flexnn/$model_name.flexnn malloc_plan_path=schedules/$model_name.flexnn.malloc layer_dependency_path=schedules/$model_name.flexnn.dependency memory_budget=$memory_budget config=flexnn_parallel num_threads=$num_threads loading_powersave=$loading_powersave computing_powersave=2 input_shape=$input_shape vocab_path=$home_dir/models/ncnn/gpt2-vocab.txt"

# level 0 only prints latency results
if [ $log_level -eq 0 ]; then
    $cmd_slice > /dev/null 2>&1
    $cmd_prof > /dev/null 2>&1
    $cmd_sched > /dev/null 2>&1
    $cmd_bench 2>&1 | grep "models/flexnn/$model_name.flexnn"
fi

# level 1 only prints offline planning logs
if [ $log_level -eq 1 ]; then
    $cmd_slice
    $cmd_prof
    $cmd_sched
    $cmd_bench > /dev/null 2>&1
fi

# level 2 prints all logs
if [ $log_level -eq 2 ]; then
    echo "======FlexNN Slicing Log======"
    $cmd_slice
    echo ""
    echo "======FlexNN Profiling Log======"
    $cmd_prof
    echo ""
    echo "======FlexNN Scheduling Log======"
    $cmd_sched
    echo ""
    echo "======FlexNN Execution Log======"
    $cmd_bench
fi
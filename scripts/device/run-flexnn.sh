# help
if [ $# -lt 5 ]; then
    echo "Usage: exec <model_name> <memory_budget> <conv_sz> <fc_sz> <num_threads> [<input_shape> <loading_powersave> <log_level> <cooling_down_duration>]"
    chmod 0777 print-help.sh
    ./print-help.sh model_name memory_budget conv_sz fc_sz num_threads input_shape loading_powersave log_level cooling_down_duration
    exit -1
fi

echo "run-flexnn.sh"

# init args
model_name=$1
memory_budget=$2
conv_sz=$3
fc_sz=$4
num_threads=$5
input_shape="[1,3,224,224]"
loading_powersave=3
log_level=0
cooling_down_duration=0

if [ $# -ge 6 ]; then
    input_shape=$6
fi

if [ $# -ge 7 ]; then
    loading_powersave=$6
fi

if [ $# -ge 8 ]; then
    log_level=$8
fi

if [ $# -ge 9 ]; then
    cooling_down_duration=$9
fi

# chmod
chmod 0777 bin/flexnnslice
chmod 0777 bin/flexnnprofile
chmod 0777 bin/flexnnschedule
chmod 0777 bin/benchflexnn

# commands to run flexnn binaries
cmd_slice="./bin/flexnnslice models/ncnn/$model_name.ncnn.param models/ncnn/$model_name.ncnn.bin models/flexnn/$model_name.flexnn.param models/flexnn/$model_name.flexnn.bin 0 $conv_sz $fc_sz"
cmd_prof="./bin/flexnnprofile models/flexnn/$model_name.flexnn memory_profile_path=profiles/$model_name.flexnn.memprof time_profile_path=profiles/$model_name.flexnn.timeprof input_shape=$input_shape num_threads=$num_threads"
cmd_sched="./bin/flexnnschedule profiles/$model_name.flexnn.memprof profiles/$model_name.flexnn.timeprof schedules/$model_name.flexnn.malloc schedules/$model_name.flexnn.dependency $memory_budget 1 results/$model_name.flexnn.layout"
cmd_bench="./bin/benchflexnn models/flexnn/$model_name.flexnn malloc_plan_path=schedules/$model_name.flexnn.malloc layer_dependency_path=schedules/$model_name.flexnn.dependency memory_budget=$memory_budget config=flexnn_parallel num_threads=$num_threads loading_powersave=$loading_powersave computing_powersave=2 input_shape=$input_shape cooling_down_duration=$cooling_down_duration"

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

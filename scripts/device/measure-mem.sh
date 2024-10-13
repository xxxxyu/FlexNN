# help
if [ $# -lt 2 ]; then
    echo "Usage: exec <test_command> <results_path> [<max_timeout_ms>]"
    exit -1
fi

echo "measure-mem.sh"

# args
test_command=$1
results_path=$2
max_timeout=1000000 # ms
if [ $# -eq 3 ]; then
    max_timeout=$3
fi

# Create the folder if it doesn't exist.
mkdir -p $(dirname $results_path)
# Init csv file.
echo "timestamp(ms),memory(kB)" > $results_path

# Make sure the binaries are executable
chmod -R 0777 bin

# Run the program in the background and mute the output
$test_command > /dev/null 2>&1 &
# $test_command &

# Get the PID of the background process
pid=$!

if [ -z "$pid" ]; then
    echo "Failed to run $test_command, exiting."
    exit 1
fi

echo "Start to measure memory."
# While elapsed time <= max_timeout
global_start_time=$(date +%s%3N)
global_end_time=$(($global_start_time + $max_timeout))
while [ $(date +%s%3N) -le $global_end_time ]; do
    # Get the actual physical memory usage using pmap
    pmap_output=$(pmap -x "$pid" 2>/dev/null | tail -n 1)

    # Extract the RSS value from the pmap output
    rss=$(echo "$pmap_output" | awk '{print $4}')

    timestamp=$(date "+%s%3N") # ms

    if [ -z "$rss" ]; then
        echo "Program has exited, stop memory measurement."
        echo "Elapsed time: $(($(date +%s%3N) - global_start_time)) ms"
        exit 0
    fi

    echo "$timestamp,$rss" >> $results_path

    # sleep for 0.01s
    sleep 0.01
done
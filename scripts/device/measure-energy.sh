# help
if [ $# -lt 2 ]; then
    echo "Usage: exec <test_command> <results_path> [<max_timeout_ms>]"
    exit -1
fi

echo "measure-energy.sh"

# args
test_command=$1
results_path=$2
max_timeout=1000000 # ms
if [ $# -eq 3 ]; then
    max_timeout=$3
fi

# Define the duration in seconds for idle power collection
idle_duration=1000 # ms

# Create the folder if it doesn't exist.
mkdir -p $(dirname $results_path)
# Init csv file.
echo "timestamp(ms),power(mW)" > $results_path

# Make sure the binaries are executable
chmod -R 0777 bin

# Function to get the current battery power
get_battery_power() {
  # The battery_path is dependent of the device
  # You might need to change this path to match your device
  local battery_path="/sys/class/power_supply/battery"

  # The units of current_now and voltage_now are uA and uV, respectively
  # On different devices, the units might be different, please check
  # On some device, current_now < 0 represents discharging, > 0 represents charging
  # On other devices, it might be the opposite, please check
  local current_now=$(cat "$battery_path/current_now")  # uA
  local voltage_now=$(cat "$battery_path/voltage_now")  # uV
  local power=$(echo "scale=4; ($current_now * $voltage_now) / -1000000000" | bc)   # mW

  echo "$power"
}

# Function to collect idle power
collect_idle_power() {
  echo "Collecting idle power for $idle_duration ms ..."
  
  local start_time=$(date +%s%3N)
  while [ "$(date +%s%3N)" -le "$((start_time + idle_duration))" ]; do
    local power=$(get_battery_power)
    local current_time=$(($(date +%s%3N) - global_start_time))
    echo "$current_time,$power" >> "$results_path"
    sleep 0.05
  done

  echo "Idle power collection completed!"
}

# Function to start the test program and monitor power
run_test() {
  echo "Running the test program..."
  
  # Run the program in the background and mute the output
  # $test_command > /dev/null 2>&1 &
  $test_command &
  
  # Get the PID of the background process
  pid=$!
  
  if [ -z "$pid" ]; then
      echo "Failed to run $test_command, exiting."
      exit 1
  fi

  local start_time=$(date +%s%3N)
  local end_time=$((start_time + max_timeout))

  while [ "$(date +%s%3N)" -le "$end_time" ]; do
    local power=$(get_battery_power)
    local current_time=$(($(date +%s%3N) - global_start_time))

    # exit if the process has terminated
    pmap_output=$(pmap -x "$pid" 2>/dev/null | tail -n 1)
    # Extract the RSS value from the pmap output
    rss=$(echo "$pmap_output" | awk '{print $4}')
    if [ -z "$rss" ]; then
        echo "Program has exited, stop memory measurement."
        break
    fi

    echo "$current_time,$power" >> "$results_path"
    sleep 0.02
  done

  echo "Test program execution completed!"
}

# Main script entry point
echo "Power measurement script started."
global_start_time=$(date +%s%3N)

collect_idle_power

run_test

collect_idle_power

echo "Power measurement completed. Power data saved to $results_path."
echo "Power measurement elapsed time: $(($(date +%s%3N) - global_start_time)) ms"

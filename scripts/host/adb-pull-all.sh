# Make sure you have Android Debug Bridge (adb) installed
# and your device is connected to your computer / docker container.

device_root_dir=/data/local/tmp/FlexNN

# Create a result directory on the device, use datetime as suffix
datetime=$(date +'%m%d%H%M')
result_dir="results_$datetime"
mkdir -p $result_dir

# Change this to the path of your adb if it is not in your PATH
# If you are using WSL, you need to call adb.exe instead of `apt install adb` in WSL
adb="adb.exe"

$adb pull $device_root_dir/results/end2end $result_dir
$adb pull $device_root_dir/results/overhead $result_dir
$adb pull $device_root_dir/results/ablation $result_dir
$adb pull $device_root_dir/results/adaption $result_dir

echo "Finished pulling all the files from the device."
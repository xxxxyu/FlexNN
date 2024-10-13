# Make sure you have Android Debug Bridge (adb) installed
# and your device is connected to your computer / docker container.

device_tmp_dir=/data/local/tmp
device_root_dir=/data/local/tmp/FlexNN

# Change this to the path of your adb if it is not in your PATH
# If you are using WSL, you need to call adb.exe instead of `apt install adb` in WSL
adb="adb.exe"

# 1. Push all the models, binaries, scripts to /data/local/tmp
$adb push build-android-aarch64/install/bin/* $device_tmp_dir
$adb push models/* $device_tmp_dir
$adb push scripts/device/* $device_tmp_dir

# 2. Setup the correct file structure on the device side
$adb shell "mkdir -p $device_root_dir && cd $device_tmp_dir && chmod 0777 setup-files.sh && ./setup-files.sh $device_tmp_dir $device_root_dir"

echo "Finished pushing all the files to the device."
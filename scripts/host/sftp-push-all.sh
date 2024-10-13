# Make sure you have established SFTP connection with SSH Key Pair Authentication.
# Note that password authentication is not supported (for executing a batch file).

# Modify this to your need
device_tmp_dir="/home/aiot/flexnn-sftp-tmp"
device_root_dir="/home/aiot/FlexNN"
batch_file="scripts/host/sftp-push-all.bat"

# SSH login information, change this to your own
# $private_key="~/.ssh/id_rsa"
username="aiot"
hostname="192.168.20.191"

# Create a temporary directory on the device
ssh $username@$hostname "mkdir -p $device_tmp_dir"

# Generate a batch file to upload all the files to the device temp directory
> $batch_file
echo "cd $device_tmp_dir" >> $batch_file
echo "put -r build-aarch64-linux-gnu/install/bin/*" >> $batch_file
echo "put -r models/*" >> $batch_file
echo "put -r scripts/device/*" >> $batch_file

# Execute the batch file
echo "sftp $username@$hostname < $batch_file"
sftp $username@$hostname < $batch_file
# sftp -i $private_key $username@$hostname < $batch_file

# Execute the setup-files.sh on the device side
ssh $username@$hostname "cd $device_tmp_dir && chmod 0777 setup-files.sh && ./setup-files.sh $device_tmp_dir $device_root_dir"

echo "Finished pushing all the files to the device."
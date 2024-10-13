# Make sure you have established SFTP connection with SSH Key Pair Authentication.
# Note that password authentication is not supported (for executing a batch file).

# Modify this to your need
device_root_dir="/home/aiot/FlexNN"
batch_file="scripts/host/sftp-pull-all.bat"

# SSH login information, change this to your own
# $private_key="~/.ssh/id_rsa"
username="aiot"
hostname="192.168.20.191"

# Create a result directory on the device, use datetime as suffix
datetime=$(date +'%m%d%H%M')
result_dir="results_$datetime"
mkdir -p $result_dir

# Generate a batch file to retrieve all the files to the local results directory
> $batch_file
echo "cd $device_root_dir/results" >> $batch_file
echo "get -r end2end $result_dir" >> $batch_file # TODO: dir name???????? go on ---
echo "get -r overhead $result_dir" >> $batch_file
echo "get -r ablation $result_dir" >> $batch_file
echo "get -r adaption $result_dir" >> $batch_file

# Execute the batch file
echo "sftp $username@$hostname < $batch_file"
sftp $username@$hostname < $batch_file
# sftp -i $private_key $username@$hostname < $batch_file

echo "Finished pulling all the files from the device."
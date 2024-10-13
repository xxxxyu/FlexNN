# Make sure you have transferred all the necessary files to the device.
# and this script is in the root of your working directory on the device.

# Read the input argument.
tmp_dir=$1
root_dir=$2

echo "Setting up root directory: $root_dir"
echo "Moving files from $tmp_dir to $root_dir"

mkdir -p $root_dir/bin
mkdir -p $root_dir/models/ncnn
mkdir -p $root_dir/models/flexnn
mkdir -p $root_dir/profiles
mkdir -p $root_dir/schedules
mkdir -p $root_dir/results

echo "Moving binaries ..."
mv $tmp_dir/flexnn* $tmp_dir/benchflexnn $root_dir/bin
echo "Moving models ..."
mv $tmp_dir/*.ncnn.* $tmp_dir/gpt2-vocab.txt $root_dir/models/ncnn
mv $tmp_dir/*.flexnn.* $root_dir/models/flexnn
echo "Moving scripts ..."
mv $tmp_dir/*.sh $root_dir

echo "Finished setting up root directory: $root_dir"
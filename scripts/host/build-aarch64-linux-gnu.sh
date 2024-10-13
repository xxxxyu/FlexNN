# Host: Linux x86_64
# Target: Linux GNU AArch64

echo "Building for Linux GNU AArch64"

# install GNU toolchain
if command -v aarch64-linux-gnu-g++ &> /dev/null
then
    echo "aarch64-linux-gnu-g++ is already installed"
else
    echo "aarch64-linux-gnu-g++ is not installed, installing..."
    apt-get update && apt-get install -y g++-aarch64-linux-gnu
fi

# Build the project
mkdir -p build-aarch64-linux-gnu
cd build-aarch64-linux-gnu
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j$(nproc)
make install
cd ..

echo "Linux GNU AArch64 build complete!"
echo "Binaries in ./build-aarch64-linux-gnu/install/bin/"
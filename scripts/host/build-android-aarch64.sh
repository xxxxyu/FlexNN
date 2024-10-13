# Host: Linux x86_64
# Target: Android AArch64

echo "Building for Android AArch64"

# Download and config NDK
# Set ANDROID_NDK=path/to/ndk if you have already installed it.
if [ -z "$ANDROID_NDK" ]; then
    echo "NDK is not installed, installing ..."
    mkdir -p tmp
    wget -O tmp/android-ndk.zip https://dl.google.com/android/repository/android-ndk-r21d-linux-x86_64.zip
    unzip tmp/android-ndk.zip -d /opt
    rm tmp/android-ndk.zip
    export ANDROID_NDK=/opt/android-ndk-r21d
    echo "NDK installed to $ANDROID_NDK"
else
    echo "NDK already installed at $ANDROID_NDK"
fi

# Remove hardcoded debug flag in Android NDK
sed -i '/-g/d' $ANDROID_NDK/build/cmake/android.toolchain.cmake # ndk<23
sed -i '/-g/d' $ANDROID_NDK/build/cmake/android-legacy.toolchain.cmake # ndk>=23

# Build the project
mkdir -p build-android-aarch64
cd build-android-aarch64

cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-21 \
    ..

# If you use cmake >= 3.21 and ndk-r23
# you need to add -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=False option for working optimization flags

make -j$(nproc)
make install
cd ..

echo "Android AArch64 build complete!"
echo "Binaries in ./build-android-aarch64/install/bin/"
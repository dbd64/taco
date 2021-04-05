export LLVM_DIR=$(brew --prefix llvm)
export LIBOMP_DIR=$(brew --prefix libomp)
export MacOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

export CPPFLAGS="-I$LLVM_DIR/include -I$LLVM_DIR/include/c++/v1/ -I$MacOS_SDK/usr/include"
export LDFLAGS="-L $LIBOMP_DIR/lib -L $MacOS_SDK/usr/lib -L/usr/lib -L$LLVM_DIR/lib -Wl,-rpath,$LLVM_DIR/lib"
export TACO_CC="$LLVM_DIR/bin/clang $CPPFLAGS $LDFLAGS"

export TACO_INCLUDE_DIR=${0:a:h}/include
export TACO_LIBRARY_DIR=${0:a:h}/cmake-build-relwithdebinfo/lib

export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$TACO_LIBRARY_DIR
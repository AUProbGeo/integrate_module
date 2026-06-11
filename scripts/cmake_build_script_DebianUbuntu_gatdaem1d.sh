#!/bin/sh
#
# This script is for compiling the forward code, gatdaem1d, from ga-aem (version >= 2.0.0), on Debian and Ubuntu
# This script has been tested and confirmed to work on both Debian and Ubuntu distributions.
# See more detailed information on https://github.com/GeoscienceAustralia/ga-aem/blob/master/cmake_build_script_ubuntu.sh
#
# Test this script with the latest Debian using a docker container:
#   docker run -it --rm -v $(pwd):/workspace -e DEBIAN_FRONTEND=noninteractive debian:latest /workspace/cmake_build_script_DebianUbuntu_gatdaem1d.sh
# Test this script with the latest Ubuntu using a docker container:
#	docker run -it --rm -v $(pwd):/workspace -e DEBIAN_FRONTEND=noninteractive ubuntu:latest /workspace/cmake_build_script_DebianUbuntu_gatdaem1d.sh
#
#
set -e  # exit with error if a command fails
set -x  # show commands while executing


# Install necessary packages in Debian/Ubuntu (tested on Ubuntu 22.04 LTS and Debian 13)
# Check if sudo is available, if not assume user is root
PREFIX=
if command -v sudo >/dev/null 2>&1; then
	PREFIX=sudo
fi
$PREFIX sh -c '
		apt-get update &&
		apt-get install -y build-essential libfftw3-dev libfftw3-bin libfftw3-double3 libopenmpi-dev cmake pkg-config git python3 python3-pip python3-venv&&
		apt-get autoremove -y'

## 1. Clone the ga-aem repository from Github
if ! test -d ga-aem
then
	git clone --recursive https://github.com/GeoscienceAustralia/ga-aem.git
fi
cd ga-aem

## 2. Compile GA-AEM

# INSTALL_DIR is the directory for installing the build package
INSTALL_DIR="$PWD"/install-ubuntu

# BUILD_DIR is a temporary directory for building (compiling and linking)
BUILD_DIR="$PWD"/build-ubuntu

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# compile gatdaem1d with optimized compiler flags
cmake -Wno-dev \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -funroll-loops" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -funroll-loops" \
  -DWITH_MPI=OFF \
  -DWITH_NETCDF=OFF \
  -DWITH_GDAL=OFF \
  -DWITH_PETSC=OFF \
  ..
cmake --build . --target matlab-bindings --config=Release
cmake --build . --target python-bindings --config=Release
cmake --install . --prefix "$INSTALL_DIR"


## CHECK FOR shared library dependencies
cd ..
readelf -d install-ubuntu/python/gatdaem1d/gatdaem1d.so  | grep 'Shared'

## 3. Install python module
echo  "installing from --> $INSTALL_DIR"
cd "$INSTALL_DIR"/python


# Check if PEP 668 is enabled (for example, in Debian/Ubuntu with system Python)
if [ -f "$(python3 -c "import sys; print(sys.prefix)")/lib/python3."*"/EXTERNALLY-MANAGED" ]; then
	    echo "✓ PEP 668 is enabled - proceeding with venv (for use with Docker)"
		python3 -m venv venv
		. venv/bin/activate
fi

# Prefer uv pip if available, fall back to pip
if command -v uv >/dev/null 2>&1; then
	echo "uv detected - using 'uv pip install'"
	if uv pip install -e .; then
		echo "Python module installed successfully"
		PYTHON_INSTALL_SUCCESS=true
	else
		echo "Python module installation failed"
		PYTHON_INSTALL_SUCCESS=false
	fi
elif command -v pip >/dev/null 2>&1; then
	if pip install -e .; then
		echo "Python module installed successfully"
		PYTHON_INSTALL_SUCCESS=true
	else
		echo "Python module installation failed"
		PYTHON_INSTALL_SUCCESS=false
	fi
else
	echo "ERROR: neither uv nor pip is installed. Please install Python and pip to complete the installation."
	echo "On Ubuntu/Debian, you can install pip with:"
	echo "  apt-get install python3 python3-pip"
	echo "Or install uv with:"
	echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
	PYTHON_INSTALL_SUCCESS=false
fi

## 4. Test the installation
if [ "$PYTHON_INSTALL_SUCCESS" = true ]; then
	echo "Running installation test..."
	python "$INSTALL_DIR"/python/examples/skytem_example.py
else
	echo "Skipping installation test due to failed python module installation"
fi

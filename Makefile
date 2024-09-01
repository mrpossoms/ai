PLATFORM := $(shell uname -s)

thirdparty:
	mkdir -p $@

thirdparty/libtorch: thirdparty
	# download libtorch for detected platform
ifeq ($(PLATFORM), Linux)
	@echo "Downloading libtorch for Linux"
	@wget -q https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip -O thirdparty/libtorch.zip
else
	@echo "Downloading libtorch for MacOS"
	@wget -q https://download.pytorch.org/libtorch/nightly/cpu/libtorch-macos-x86_64-latest.zip -O thirdparty/libtorch.zip
endif

setup: thirdparty/libtorch
	@echo "Setup complete"

.PHONY: setup
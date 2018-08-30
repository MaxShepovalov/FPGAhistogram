COMMON_REPO := ../../..

#Common Includes
include $(COMMON_REPO)/utility/boards.mk
# include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/xcl/xcl.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# Host Application
# fpga_test_SRCS=./src/host.cpp $(oclHelper_SRCS) $(xcl2_SRCS)
# fpga_test_HDRS=$(xcl2_HDRS)
# fpga_test_CXXFLAGS=-I./src/ $(opencl_CXXFLAGS) $(xcl2_CXXFLAGS) $(oclHelper_CXXFLAGS) -std=c++0x
# fpga_test_LDFLAGS=$(opencl_LDFLAGS)

fpga_test_SRCS=./src/host.cpp $(xcl_SRCS)
fpga_test_HDRS=$(xcl_HDRS)
fpga_test_CXXFLAGS=-I./src/ $(opencl_CXXFLAGS) $(xcl_CXXFLAGS) -std=c++0x
fpga_test_LDFLAGS=$(opencl_LDFLAGS)

EXES=fpga_test

# Kernel
fpga64_SRCS=./src/fpga64.cl

XOS=fpga64

fpga64_XOS=fpga64

XCLBINS=fpga64

# check
check_EXE=fpga_test
check_XCLBINS=fpga64

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk

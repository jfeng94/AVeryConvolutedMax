CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# NVCC
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode 
arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

all: iotest raytrace

iotest:
	$(CC) iotest.cpp -o iotest

raytrace.o: raytrace_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) -c raytrace_cuda.cu

camera.o: camera.cpp
	$(CC) -I$(CUDA_INC_PATH) camera.cpp -c

point.o: point.cpp
	$(CC) -I$(CUDA_INC_PATH) point.cpp -c

superquadric.o: superquadric.cpp
	$(CC) -I$(CUDA_INC_PATH) superquadric.cpp -c

matrix.o: matrix.cpp
	$(CC) -I$(CUDA_INC_PATH) matrix.cpp -c

cameratest.o: cameratest.cpp
	$(CC) -I$(CUDA_INC_PATH) cameratest.cpp -c

raytrace: cameratest.o matrix.o superquadric.o point.o camera.o raytrace.o
	$(NVCC) $(GENCODE_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) cameratest.o matrix.o superquadric.o point.o camera.o raytrace.o -o cameratest

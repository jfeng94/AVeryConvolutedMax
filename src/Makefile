CC = /usr/bin/g++

LD_FLAGS = -lrt

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
ifneq ($(DARWIN),)
      CUDA_PATH ?= /usr/local/cuda-6.5
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
      CUDA_PATH ?= /Developer/NVIDIA/CUDA-7.0
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# NVCC
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

all: raytrace

iotest: matrix.o superquadric.o point.o 
	$(CC) -I$(CUDA_INC_PATH) $(LDFLAGS) iotest.cpp matrix.o superquadric.o point.o -o iotest

raytrace.o: raytrace_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info raytrace_cuda.cu

camera.o: camera.cpp
	$(CC) -I$(CUDA_INC_PATH) -c camera.cpp 

parser.o: parser.cpp
	$(CC) -I$(CUDA_INC_PATH) -c parser.cpp 

point.o: point.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info point.cu

superquadric.o: superquadric.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info superquadric.cu 

matrix.o: matrix.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info matrix.cu

cameratest.o: cameratest.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) -rdc=true -I$(CUDA_INC_PATH) -c -line_info cameratest.cu 

raytrace: cameratest.o matrix.o superquadric.o point.o parser.o camera.o raytrace.o
	$(NVCC) $(GENCODE_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) -I$(CUDA_INC_PATH) -DTHRUST_DEBUG cameratest.o matrix.o superquadric.o point.o parser.o camera.o raytrace_cuda.o -o cameratest

clean:
	rm -f *.o cameratest iotest

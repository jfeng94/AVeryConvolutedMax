#ifndef CUDA_RAYTRACE_CUH
#define CUDA_RAYTRACE_CUH
#include <thrust/device_vector.h>
#include "superquadric.cuh"
#include "point.cuh"

void cudaCallScenePrep(thrust::device_vector<Superquadric> scene, unsigned int size,
                       int blocks, int threadsPerBlock);

void cudaCallRayTrace(Superquadric * object,
                  thrust::device_vector<Superquadric> scene, 
                  thrust::device_vector<pointLight> lights,
                  thrust::device_vector<Ray> screen,
                  unsigned int size, Point lookFrom, int blocks,
                  int threadsPerBlock);

#endif //CUDA_RAYTRACE_CUH

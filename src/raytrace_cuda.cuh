#ifndef CUDA_RAYTRACE_CUH
#define CUDA_RAYTRACE_CUH
#include <thrust/device_vector.h>
#include "superquadric.cuh"
#include "point.cuh"

void cudaCallScenePrep(thrust::device_vector<Superquadric> scene, Superquadric * dev_out_scene,
	               unsigned int size,
                       int blocks, int threadsPerBlock);

void cudaCallRayTrace(Superquadric * object,
                  thrust::device_vector<Superquadric> scene, 
                  thrust::device_vector<pointLight> lights,
                  Ray * RayScreen,
                  unsigned int size, Point * lookFrom, int blocks,
                  int threadsPerBlock);

#endif //CUDA_RAYTRACE_CUH

#ifndef CUDA_RAYTRACE_CUH
#define CUDA_RAYTRACE_CUH


void cudaCallScenePrep(thrust::device_vector<Superquadric> scene, int size,
                       int blocks, int threadsPerBlock);

void cudaCallRayTrace(Superquadric object,
                  thrust::device_vector<Superquadric> scene, 
                  thrust::device_vector<pointLight> lights,
                  thrust::device_vector<Ray> screen,
                  int size, Point * lookFrom, int blocks,
                  int threadsPerBlock);

#endif //CUDA_RAYTRACE_CUH
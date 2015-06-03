#include <cuda_runtime.h>
#include <cuda.h>
#include "raytrace_cuda.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "superquadric.h"
#include "point.h"
#include "matrix.h"
#include "camera.h"
#include <iostream>
#include <math.h>


// This kernel will parallelize the scene preparation
__global__
void cudaScenePrep(thrust::device_vector<Superquadric> scene, int size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Thread Resiliency
    while (index < size) {
        scene[index].setNum(index);
        index += blockDim.x * gridDim.x;
    }
    // Syncing threads so that they all finish.
    __syncthreads();
}


// This will just call the kernel...
void cudaCallScenePrep(thrust::device_vector<Superquadric> scene, int size,
                       int blocks, int threadsPerBlock) {
    cudaScenePrep<<<blocks, threadsPerBlock>>>(scene, size);
}


// This kernel will be called in the "runRayTrace" thing from camera.
// This will be parallelized based on the screen.
__global__
void cudaRayTrace(Superquadric object,
                  thrust::device_vector<Superquadric> scene, 
                  thrust::device_vector<pointLight> lights,
                  thrust::device_vector<Ray> screen,
                  int size, Point * lookFrom) {
    // Thread resiliency measures.
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size) {
        Ray targetRay = screen[index];
        Point * origin = targetRay.getStart();
        Point * dir = targetRay.getDir();

        // Transform frame of reference so that this object is at origin.
        origin = object.applyTransforms(origin);
        dir = (object.applyDirTransforms(dir))->norm();

        // Create new ray to do intersection test.
        Ray transR;
        transR.setStart(origin);
        transR.setDir(dir);

        // Check for intersection
        float intersects = get_intersection(transR);

        // If there is an intersection
        if (intersects != lFLT_MAX && intersects < r.getTime()) {
            // Calculate the intersection point
            Point * pTran = transR.propogate(intersects);
            Point * pTrue = object.revertTransforms(pTran);

            // Get the normal at the intersection point
            Point * n = object.revertDirTransforms((object.getNormal(pTran))->norm());

            Point *showNorm = *pTran + *(*n / 10);

            Point * color = lighting(pTrue, n, lookFrom, lights, scene);

            r.SetColor(color->X(), color->Y(), color->Z());
        }
        index += blockDim.x * gridDim.x;
    }
    // Syncing threads so that they all finish...
    __syncthreads();
}

void cudaCallRayTrace(Superquadric object,
                      thrust::device_vector<Superquadric> scene, 
                      thrust::device_vector<pointLight> lights,
                      thrust::device_vector<Ray> screen,
                      int size, Point * lookFrom, int blocks,
                      int threadsPerBlock) {


    cudaRayTrace<<<blocks, threadsPerBlock>>>
                (object, scene, lights, screen, size, lookFrom);
}


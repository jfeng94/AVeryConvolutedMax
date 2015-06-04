#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include "raytrace_cuda.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "superquadric.cuh"
#include "point.cuh"
#include "matrix.cuh"
#include "camera.h"
#include <iostream>
#include <math.h>
#include <float.h>

// This kernel will parallelize the scene preparation
__global__
void cudaScenePrep(Superquadric * start, unsigned int size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Thread Resiliency
    while (index < size) {
        (*(start + index)).setNum(index);
        index += blockDim.x * gridDim.x;
    }
    // Syncing threads so that they all finish.
    __syncthreads();
}


// This will just call the kernel...
void cudaCallScenePrep(thrust::device_vector<Superquadric> scene, unsigned int size,
                       int blocks, int threadsPerBlock) {

    Superquadric * start = thrust::raw_pointer_cast(&scene[0]);
    cudaScenePrep<<<blocks, threadsPerBlock>>>(start, size);
}


// This kernel will be called in the "runRayTrace" thing from camera.
// This will be parallelized based on the screen.
__global__
void cudaRayTrace(Superquadric * object,
                  Superquadric * sceneStart, 
                  pointLight * lightStart,
                  Ray * start,
                  unsigned int size, unsigned int lightSize, unsigned int sceneSize, Point * lookFrom) {
    // Thread resiliency measuresi.
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size) {
        //WAHT IS GOING ON HERE AEBRGUKAEGBKAERBAG
        // WHY DO WE KEEP GETTING ADDRESS 0X00000000 AT GET_INTERSECTIONS?
        // I DELETED SO MANY THINGS MANANGRUYGBAKRUGBAEGB
        printf("eccentricity = %f.\n", object->getEcc());
        Ray targetRay = *(start + index);
        Point origin = targetRay.getStart();
        Point dir = targetRay.getDir();

        Point temp_lookFrom;
        temp_lookFrom.set(lookFrom->X(), lookFrom->Y(), lookFrom->Z());

        // Transform frame of reference so that this object is at origin.
        Point new_origin = object->applyTransforms(origin);
        Point new_dir = (object->applyDirTransforms(dir)).norm();


        // Create new ray to do intersection test.
        Ray transR;
        transR.setStart(new_origin);
        transR.setDir(new_dir);

        // Check for intersection
        float intersects = object->get_intersection(transR);
        // If there is an intersection
        if (intersects != FLT_MAX && intersects < targetRay.getTime()) {
            // Calculate the intersection point
            Point pTran = transR.propagate(intersects);
            Point pTrue = object->revertTransforms(pTran); 
            // Get the normal at the intersection point
            Point n = object->revertDirTransforms((object->getNormal(pTran)).norm());
            // Point *showNorm = *pTran + *(*n / 10);
            Point color = object->lighting(pTrue, n, temp_lookFrom, lightStart, sceneStart,
                                            lightSize, sceneSize);

            targetRay.setColor(color.X(), color.Y(), color.Z());
           }
        index += blockDim.x * gridDim.x;
    } 
    // Syncing threads so that they all finish...
    __syncthreads();
}

void cudaCallRayTrace(Superquadric * object,
                      thrust::device_vector<Superquadric> scene, 
                      thrust::device_vector<pointLight> lights,
                      thrust::device_vector<Ray> screen,
                      unsigned int size, Point * lookFrom, int blocks,
                      int threadsPerBlock) {
    std::cout << "Calling it now" << std::endl;
    Ray * start = thrust::raw_pointer_cast(&screen[0]);
    pointLight * lightStart = thrust::raw_pointer_cast(&lights[0]);
    Superquadric * sceneStart = thrust::raw_pointer_cast(&scene[0]);
    std::cout << "Got the pointers done" << std::endl;

    unsigned int lightSize = lights.size();
    unsigned int sceneSize = scene.size();
    std::cout << "Sizes ready!" << std::endl;

    cudaRayTrace<<<blocks, threadsPerBlock>>>
                (object, sceneStart, lightStart, start, size, lightSize,
                sceneSize, lookFrom);
}


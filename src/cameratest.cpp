#include "superquadric.h"
#include "point.h"
#include "matrix.h"
#include "camera.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include "raytrace_cuda.cuh"
#include <iostream>
#include <string>

int main(int argc, char ** argv)
{
    Point *rot, *tra, *sca, *dif, *amb, *spe;
    float theta, e, n, shi, sne, opa;
    
    rot               = new Point(1, 0, 0);
    tra               = new Point(-1, -5, 1);
    sca               = new Point(1, 1, 1);
    theta             = 0;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s1 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = new Point(0, 0, 1);
    tra               = new Point(-3, -5, -1);
    sca               = new Point(1, 1, 1);
    theta             = 3.1415926 / 4;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = new Point(0, 0, 1);
    tra               = new Point(-3, -5, 3);
    sca               = new Point(1, 1, 1);
    theta             = 3.1415926 / 4;
    e                 = 1;
    n                 = 1;
    Superquadric * s3 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = new Point(0, 0, 1);
    tra               = new Point(-1, -5, -3);
    sca               = new Point(1, 1, 1);
    theta             = 3.1415926 / 4;
    e                 = 1;
    n                 = 1;
    Superquadric * s4 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = new Point(0, 0, 1);
    tra               = new Point(-20, -50, 0);
    sca               = new Point(28, 28, 28);
    dif               = new Point( 10, 140, 125);
    amb               = new Point(130, 130, 130);
    spe               = new Point(150, 150, 150);
    shi               = 0.01;
    theta             = 3.1415926 / 8;
    e                 = 2;
    n                 = 2;
    Superquadric * s5 = new Superquadric(tra, sca, rot, theta, e, n,
                                         dif, amb, spe, shi, sne, opa);

    // Preparing for CPU stuff
    std::cout << "Preparing for CPU Raytracing..." << std::endl;
    std::vector<Superquadric> scene;
    scene.push_back(*s5);
    scene.push_back(*s1);
    scene.push_back(*s2);
    scene.push_back(*s3);
    scene.push_back(*s4);


    pointLight *l1 = new pointLight( 0, 15, 0,   0, 140, 125, 0.005);
    pointLight *l2 = new pointLight( 0,-15, 0, 140,   0, 125, 0.005);
    pointLight *l3 = new pointLight( 2, 5, 0, 255, 255, 255, 10);
    pointLight *l4 = new pointLight( 5, 5,-5, 125, 140,   0, 0.005);
    std::vector<pointLight> lights;
    lights.push_back(*l1);
    lights.push_back(*l2);
    lights.push_back(*l3);
    lights.push_back(*l4);



    Point * LookFrom = new Point(2, 5, 0);
    Point * LookAt   = new Point(0, 0, 0);
    Point * Up       = new Point(0, 0, 1);
    float Fd         = 0.05;
    float Fx         = 0.08;
    float Nx         = 1920;
    float Ny         = 1080;
    Camera *c = new Camera(LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);

    std::cout << "Raytracing..." << std::endl;
    c->runRayTracer(scene, lights);

    std::cout << "Printing..." << std::endl;
    c->printImage();

    std::cout << "CPU RayTracing done!" << std::endl;

    // Now, for GPU implementation
    if (argc != 3) {
        std::cout << "For GPU usage: ./cameratest <numBlocks> <threadsPerBlock>" << std::endl;
        return 0;
    }

    std::cout << "Preparing for GPU Raytracing..." << std::endl;
    // First, get number of blocks
    int blocks = stoi(argv[1]);

    // Then, threadsPerBlock
    int threadsPerBlock = stoid(argv[2]);

    // Create a new camera with the same things as above.
    Camera * d_c = new Camera(LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);

    // Create two device_vectors from the std::vectors above.
    thrust::device_vector<Superquadric> d_scene(scene.begin(), scene.end());
    thrust::device_vector<pointLight> d_lights(lights.begin(), lights.end());

    // Create a device_vector based on the screen from the camera.
    thrust::device_vector<Ray> d_screen(d_c->rayScreen.begin(), d_c->rayScreen.end());

    // Get size values for the thread resiliency...
    int d_scene_size = d_scene.size();
    int d_lights_size = d_lights.size();
    int d_screen_size = d_screen.size();

    // Prepare the scene...
    cudaCallScenePrep(d_scene, d_scene_size, blocks, threadsPerBlock);

    // Running the Ray Tracer...

    // Adding an eye light
    pointLight *d_l = new pointLight(this->LookFrom.X(),
                                     this->LookFrom.Y(),
                                     this->LookFrom.Z(),
                                     255, 255, 255, 1);
    d_lights.push_back(*d_l);

    std::cout << "Raytracing..." << std::endl;
    // Allocate space for the point on the GPU
    Point * d_lookFrom;
    cudaMalloc(&d_lookFrom, sizeof(Point));
    cudaMemcpy(d_lookFrom, &d_c->LookFrom, sizeof(Point), cudaMemcpyHostToDevice);

    for(int i = 0; i < d_scene_size; i++) {
        Superquadric object = d_scene[i];
        cudaCallRayTrace(object, d_scene, d_lights, d_screen, d_screen_size, 
                         d_lookFrom, blocks, threadsPerBlock);
    }


    // The screen is done. Set the camera's ray vector to be equal to the 
    // screen thrust::vector.
    std::vector<Ray> out_screen;
    thrust::copy(d_screen.begin(), d_screen.end(), out_screen.begin());
    d_c->rayScreen = out_screen;


    std::cout << "Printing..." << std::endl;
    // Now, print that puppy out.
    d_c->gpuPrintImage();

    std::cout << "GPU RayTracing done!" << std::endl;

    // Free all the things.
    free(c);
    free(d_c);
    cudaFree(d_lookFrom);

    // Thrust vectors automatically freed upon returning.
    return 0;
}

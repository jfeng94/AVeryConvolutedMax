#include "superquadric.cuh"
#include "point.cuh"
#include "matrix.cuh"
#include "camera.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include "raytrace_cuda.cuh"
#include <iostream>

// Timing setup Code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {              \
    cudaEventCreate(&start);         \
    cudaEventCreate(&stop);          \
    cudaEventRecord(start);          \
}

#define STOP_RECORD_TIMER(name) {    \
    cudaEventRecord(stop);           \
    cudaEventSynchronize(stop);      \
    cudaEventElapsedTime(&name, start, stop); \
    cudaEventDestroy(start);         \
    cudaEventDestroy(stop);          \
}


int main(int argc, char ** argv)
{

    Point rot, tra, sca, dif, amb, spe;
    float theta, e, n, shi, sne, opa;
    
    rot               = Point(1, 0, 0);
    tra               = Point(-1, -5, 1);
    sca               = Point(1, 1, 1);
    theta             = 0;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s1 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = Point(0, 0, 1);
    tra               = Point(-3, -5, -1);
    sca               = Point(1, 1, 1);
    theta             = 3.1415926 / 4;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = Point(0, 0, 1);
    tra               = Point(-3, -5, 3);
    sca               = Point(1, 1, 1);
    theta             = 3.1415926 / 4;
    e                 = 1;
    n                 = 1;
    Superquadric * s3 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = Point(0, 0, 1);
    tra               = Point(-1, -5, -3);
    sca               = Point(1, 1, 1);
    theta             = 3.1415926 / 4;
    e                 = 1;
    n                 = 1;
    Superquadric * s4 = new Superquadric(tra, sca, rot, theta, e, n);

    rot               = Point(0, 0, 1);
    tra               = Point(-20, -50, 0);
    sca               = Point(28, 28, 28);
    dif               = Point( 10, 140, 125);
    amb               = Point(130, 130, 130);
    spe               = Point(150, 150, 150);
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
    Point LookAt   = Point(0, 0, 0);
    Point Up       = Point(0, 0, 1);
    float Fd         = 0.05;
    float Fx         = 0.08;
    float Nx         = 1920;
    float Ny         = 1080;
    Camera *c = new Camera(*LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);

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
    int blocks = atoi(argv[1]);

    // Then, threadsPerBlock
    int threadsPerBlock = atoi(argv[2]);

    // Create a new camera with the same things as above.
    Camera * d_c = new Camera(*LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);
    
    // Create two device_vectors from the std::vectors above.
    thrust::device_vector<Superquadric> d_scene(scene.begin(), scene.end());
    thrust::device_vector<pointLight> d_lights(lights.begin(), lights.end());


    // Create a device_vector based on the screen from the camera.
    std::vector<Ray> camScreen = d_c->getRayScreen();
    thrust::device_vector<Ray> d_screen(camScreen.begin(), camScreen.end());

    // Get size values for the thread resiliency...
    unsigned int d_scene_size = d_scene.size();
    unsigned int d_lights_size = d_lights.size();
    unsigned int d_screen_size = d_screen.size();

    // Allocate space for the out_scene.
    Superquadric * dev_out_scene;
    cudaMalloc(&dev_out_scene, sizeof(Superquadric) * d_scene_size);
    
    // Prepare the scene...
    cudaCallScenePrep(d_scene, dev_out_scene, d_scene_size, blocks, threadsPerBlock);
    
    std::cout << "Scene Done Being Prepared" << std::endl;
    // Running the Ray Tracer...

    // Adding an eye light
    pointLight *d_l = new pointLight(LookFrom->X(),
                                     LookFrom->Y(),
                                     LookFrom->Z(),
                                     255, 255, 255, 1);
    d_lights.push_back(*d_l);

    std::cout << "Raytracing..." << std::endl;
    // Allocate space for the point on the GPU
    Point * d_lookFrom;
    cudaMalloc(&d_lookFrom, sizeof(Point));
    cudaMemcpy(d_lookFrom, LookFrom, sizeof(Point), cudaMemcpyHostToDevice);

    Ray * RayScreen;
    cudaMalloc(&RayScreen, sizeof(Ray) * d_screen_size);
    Ray * dev_vector_start = thrust::raw_pointer_cast(&d_screen[0]);
    cudaMemcpy(RayScreen, dev_vector_start, sizeof(Ray) * d_screen_size, cudaMemcpyDeviceToDevice);

    for(int i = 0; i < d_scene_size; i++) {
        cudaCallRayTrace(dev_out_scene + i, d_scene, d_lights, RayScreen, d_screen_size, 
                         d_lookFrom, blocks, threadsPerBlock);
    }
	std::cout << "Done with raytrace..." << std::endl;


    // The screen is done. Set the camera's ray vector to be equal to the 
    // screen thrust::vector.
    Ray * host_Screen;
    host_Screen = (Ray*) malloc(sizeof(Ray) * d_screen_size);
    cudaMemcpy(host_Screen, RayScreen, sizeof(Ray) * d_screen_size, cudaMemcpyDeviceToHost);

    std::vector<Ray> out_screen(host_Screen, host_Screen + d_screen_size);
    d_c->setRayScreen(out_screen);


    std::cout << "Printing..." << std::endl;
    // Now, print that puppy out.
    d_c->gpuPrintImage();

    std::cout << "GPU RayTracing done!" << std::endl;

    // Free all the things.
    delete c;
    delete d_c;
    free(host_Screen);
    cudaFree(dev_out_scene);
    cudaFree(d_lookFrom);
    cudaFree(RayScreen);

    // Thrust vectors automatically freed upon returning.
    return 0;
}

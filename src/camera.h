#ifndef CAMERA_H
#define CAMERA_H

#include "point.h"
#include "superquadric.h"
#include "matrix.h"
#include <vector>

// TODO When ready, add Thrust library, convert vector to thrust vectors
class Camera
{
    private:
        std::vector<Ray> rayScreen;
        Point LookFrom, LookAt, Up;
        Point e1, e2, e3;
        float Fd, Fx, Fy;
        int Nx, Ny;

        void init();
    public:
        // Default constructor
        Camera();

        // Camera constructor
        //     LookFrom LookAt   Fd     Fx     Nx     Ny
        Camera(Point *, Point *, Point *, float, float, float, float);
        void runRayTracer(std::vector<Superquadric>, std::vector<pointLight>);

        void scenePrep(std::vector<Superquadric>);
        void printImage();
};

#endif

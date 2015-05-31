#ifndef CAMERA_H_
#define CAMERA_H_

#include "point.h"
#include "superquadric.h"
#include "matrix."
#include <vector>

class Camera
{
    private:
        std::vector<std::vector<Ray *>> rayScreen;
        Point LookFrom, LookAt, Up;
        Point e1, e2, e3;
        float Fd;
        int Fx, Fy, Nx, Ny;

        Camera();

        void init();
    public:
        // Default constructor
        Camera();

        // Camera constructor
        //     LookFrom LookAt   Fd     Fx     Nx     Ny
        Camera(Point *, Point *, Point *, float, float, float, float);
        RayTrace(std::vector<Superquadric *>);

};
#endif // CAMERA_H_

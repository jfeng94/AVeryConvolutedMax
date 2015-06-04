#include "superquadric.h"
#include "point.h"
#include "matrix.h"
#include "camera.h"
#include "parser.h"

int main(int argc, char ** argv)
{
    // Argv[1] will be scene file
    std::vector<Superquadric> scene = parseObjects(argv[1]);

    std::vector<pointLight> lights;
    pointLight *l = new pointLight(12, 5, 0, 255, 255, 255, 0.004);
    lights.push_back(*l);


    Point * LookFrom = new Point(18, 5, 15);
    Point * LookAt   = new Point(0, 0, 0);
    Point * Up       = new Point(0, 0, 1);
    float Fd         = 0.05;
    float Fx         = 0.1;
    float Nx         = 2880;
    float Ny         = 1800;
    Camera *c = new Camera(LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);
    c->runRayTracer(scene, lights);
    c->printImage();
}

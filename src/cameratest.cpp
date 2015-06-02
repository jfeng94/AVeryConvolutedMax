#include "superquadric.cpp"
#include "point.cpp"
#include "matrix.cpp"
#include "camera.cpp"

int main()
{
    Point *rot, *tra, *sca;
    float theta, e, n;
    
    rot              = new Point(1, 0, 0);
    tra              = new Point(0, 0, -1);
    sca              = new Point(1, 1, 1);
    theta             = 0;
    e                 = 1;
    n                 = 1;
    Superquadric * s1 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(1, 0, 0);
    tra              = new Point(0, 0, 1);
    sca              = new Point(1, 1, 1);
    theta             = 0;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    std::vector<Superquadric> scene;
    scene.push_back(*s1);
    scene.push_back(*s2);

    pointLight l;
    std::vector<pointLight> lights;
    lights.push_back(l);

    Point * LookFrom = new Point(2, 5, 0);
    Point * LookAt   = new Point(0, 0, 0);
    Point * Up       = new Point(0, 0, 1);
    float Fd         = 0.05;
    float Fx         = 0.035;
    float Nx         = 60;
    float Ny         = 60;
    Camera *c = new Camera(LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);
    c->runRayTracer(scene, lights);
    c->printImage();
    std::cout << l.getPos();
}

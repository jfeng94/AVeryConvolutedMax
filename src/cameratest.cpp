#include "superquadric.cpp"
#include "point.cpp"
#include "matrix.cpp"
#include "camera.cpp"

int main()
{
    Point *rot, *tra, *sca;
    float theta, e, n;
    
    rot              = new Point(1, 0, 0);
    tra              = new Point(-3, -5, -1);
    sca              = new Point(1, 1, 1);
    theta            = 0;
    e                = .1;
    n                = .1;
    Superquadric * s1 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(0, 0, 1);
    tra              = new Point(-1, -5, 1);
    sca              = new Point(1, 1, 1);
    theta            = 3.1415926 / 4;
    e                = 0.1;
    n                = 0.1;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(0, 0, 1);
    tra              = new Point(-3, -5, 3);
    sca              = new Point(1, 1, 1);
    theta            = 3.1415926 / 4;
    e                = 1;
    n                = 1;
    Superquadric * s3 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(0, 0, 1);
    tra              = new Point(-1, -5, -3);
    sca              = new Point(1, 1, 1);
    theta            = 3.1415926 / 4;
    e                = 1;
    n                = 1;
    Superquadric * s4 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(0, 0, 1);
    tra              = new Point(0, 0, 0);
    sca              = new Point(1, 1, 1);
    theta            = 3.1415926 / 4;
    e                = 2;
    n                = 2;
    Superquadric * s5 = new Superquadric(tra, sca, rot, theta, e, n);

    std::vector<Superquadric> scene;
    scene.push_back(*s1);
    scene.push_back(*s2);
    scene.push_back(*s3);
    scene.push_back(*s4);
    scene.push_back(*s5);


    pointLight *l1 = new pointLight( 5, 5, 5,   0, 140, 125, 0.005);
    pointLight *l2 = new pointLight(-5, 5, 5, 140,   0, 125, 0.005);
    pointLight *l3 = new pointLight( 2, 5, 0, 255, 255, 255, 10);
    std::vector<pointLight> lights;
    lights.push_back(*l1);
    lights.push_back(*l2);
    lights.push_back(*l3);

    Point * LookFrom = new Point(2, 5, 0);
    Point * LookAt   = new Point(0, 0, 0);
    Point * Up       = new Point(0, 0, 1);
    float Fd         = 0.05;
    float Fx         = 0.035;
    float Nx         = 640;
    float Ny         = 1134;
    Camera *c = new Camera(LookFrom, LookAt, Up, Fd, Fx, Nx, Ny);
    c->runRayTracer(scene, lights);
    c->printImage();
}

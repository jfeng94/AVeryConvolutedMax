#include "superquadric.cpp"
#include "point.cpp"
#include "matrix.cpp"
#include "camera.cpp"

int main()
{
    Point *rot, *tra, *sca;
    float theta, e, n;
    
    Superquadric s;

    rot              = new Point(1, 0, 0);
    tra              = new Point(0, 0, 0);
    sca              = new Point(1, 1, 1);
    theta             = 0;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    std::vector<Superquadric> scene;
    scene.push_back(s);
    scene.push_back(*s2);

    Camera *c = new Camera();
    c->runRayTracer(scene);
    c->printImage();
}

#include "camera.h"
#include "point.h"
#include "superquadric.h"
#include "matrix.h"

#include <fstream>

Camera::Camera()
{
    this->LookAt   = * (new Point());
    this->LookFrom = * (new Point(5, 5, 5));
    this->Up       = * (new Point(0, 0, 1));
    this->Fd       = 0.050;
    this->Fx       = 0.035;
    this->Nx       = 400;
    this->Ny       = 400;

    // INitialize other members
    this->init();
}

Camera::Camera(Point * LookFrom, Point * LookAt, Point * Up,
               float Fd, float Fx, float Nx, float Ny)
{
    // Set given values
    this->LookFrom = *LookFrom;
    this->LookAt   = *LookAt;
    this->Up       = *Up->norm();
    this->Fd       = Fd;
    this->Fx       = Fx;
    this->Nx       = Nx;
    this->Ny       = Ny;

    // Initialize the other members of the camera
    this->init();
}

void Camera::init()
{
    // Solve for film height
    this->Fy       = (Fx / Nx) * Ny;

    float dFx = Fx / Nx;
    float dFy = Fy / Ny;

    // Get the directional vector for the camera
    Point * A = (this->LookFrom - this->LookAt)->norm();
    
    // Project A onto our Upwards vector
    float alpha = A->dot(&(this->Up)) / A->dot(A);
    Point * B = (this->Up - *(*A * alpha))->norm();

    // Get the orthogonal vector to A and B
    Point * C = (A->cross(B))->norm();

    // Get incremental vectors
    Point *DFx = *C * dFx;
    Point *DFy = *B * dFy;

    // Get upper left corner
    Point * Start = *(*A * Fd) + *(*(*DFy * (1.0 * Ny / 2)) - *(*DFx * (1.0 * Nx / 2)));
    Point * End;
    Point * Dir;

    std::ofstream out;
    out.open("MatlabTools/CameraRays.txt", std::fstream::out);
    out << &(this->LookFrom);

    // Initiate camera rays
    for (int y = 0; y < this->Ny; y++)
    {
        for (int x = 0; x < this->Nx; x++)
        {
            float px = (x * dFx) - (Fx / (double) 2);
            float py = (y * dFy) - (Fy / (double) 2);
            //End = *(*Start + *(*DFx * x)) + *(*DFy * y);
            End = *(this->LookFrom - *(*A * Fd)) + *(*(*C * px) + *(*B * py));
            out << End;
            Dir = *End - (this->LookFrom);
            Ray r;
            r.setStart(&(this->LookFrom));
            r.setDir(Dir); 

           this->rayScreen.push_back(r); 
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// RAY TRACER! WE NEED TO KERNELIZE THIS!
///////////////////////////////////////////////////////////////////////////////
void Camera::runRayTracer(std::vector<Superquadric> scene,
                          std::vector<pointLight> lights)
{
    // Flush analytical output data
    std::ofstream out;
    out.open("MatlabTools/TestRay.txt", std::fstream::out);
    out.close();
    out.open("MatlabTools/TestNormals.txt", std::fstream::out);
    out.close();
    for (int i = 0; i < scene.size(); i++)
    {
        for (int px = 0; px < this->rayScreen.size(); px++)
        {
            //std::cout << "\n";
            //std::cout << "Tracing pixel: (" << px / this->Nx << "," << px % Nx << ")\n";
            //std::cout << "Point " << this->rayScreen[px].getStart();
            scene[i].rayTrace(this->rayScreen[px], &this->LookFrom, lights);
        }
    }
}

void Camera::printImage()
{
    std::ofstream out;
    out.open("RESULT.png", std::fstream::out);

    out << "P3\n" << this->Nx << " " << this->Ny << "\n255\n";
    for (int y = this->Ny; y > 0; y--)
    {
        for (int x = this->Nx; x > 0; x--)
        {
            out << this->rayScreen[y * Nx + x].getR() << " " <<
                   this->rayScreen[y * Nx + x].getG() << " " <<
                   this->rayScreen[y * Nx + x].getB() << "\n";
        }
    }
}


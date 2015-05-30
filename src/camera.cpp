#include "point.cpp"
#include "superquadric.cpp"
#include "matrix.cpp"

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
}

#include "superquadric.h"
#include "matrix.h"
#include "point.h"

#include <math.h>

// Create a sphere at the origin of the scene.
Superquadric::Superquadric()
{
    this->t = new traMat();
    this->r = new rotMat();
    this->s = new scaMat();
    this->e = 1;
    this->n = 1;
}


// Create an object with specified eccentricities at the origin.
Superquadric::Superquadric(float E, float N)
{
    this->t = new traMat();
    this->r = new rotMat();
    this->s = new scaMat();
    this->e = E;
    this->n = N;
}

// Fully customized superquadric constructor.
Superquadric::Superquadric(Point * tra, Point * sca, Point * rot,
                           float E, float N)
{
    this->t = new traMat(tra);
    this->s = new scaMat(sca);
    this->r = new rotMat(rot, 0);
    this->e = E;
    this->n = N;
}

// Check if a given point is inside or outside of the 
float Superquadric::isq(Point *p)
{
    return pow(pow(p->X(), 2 / this->e) + pow(p->Y(), 2 / this->e),
               this->e / this->n) +
           pow(p->Z(), 2 / this->n) - 1;
}

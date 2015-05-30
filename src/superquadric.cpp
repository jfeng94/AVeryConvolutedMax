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
                           float theta, float E, float N)
{
    this->t = new traMat(tra);
    this->s = new scaMat(sca);
    this->r = new rotMat(rot, theta);
    this->e = E;
    this->n = N;
}

Point * Superquadric::applyTransforms(Point * p)
{
    Point * res = this->s->unapply(this->r->unapply(this->t->unapply(p)));

    return res;
}

// Check if a given point is inside or outside of the 
float Superquadric::isq(Point *p)
{
    Point * transP = this->applyTransforms(p);
    return pow(pow(transP->X() * transP->X(), 1 / this->e) +
               pow(transP->Y() * transP->Y(), 1 / this->e),
               this->e / this->n) +
           pow(transP->Z() * transP->Z(), 1 / this->n) - 1;
}

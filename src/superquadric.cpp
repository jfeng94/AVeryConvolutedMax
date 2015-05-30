#include "superquadric.h"
#include "matrix.h"
#include "point.h"

#include <math.h>
#include <iostream>

// Create a sphere at the origin of the scene.
Superquadric::Superquadric()
{
    this->e = 1;
    this->n = 1;
}


// Create an object with specified eccentricities at the origin.
Superquadric::Superquadric(float E, float N)
{
    this->e = E;
    this->n = N;
}

// Fully customized superquadric constructor.
Superquadric::Superquadric(Point * tra, Point * sca, Point * rot,
                           float theta, float E, float N)
    {
    this->t.set(tra);
    this->s.set(sca);
    this->r.set(rot->norm());
    this->r.setTheta(theta);
    this->e = E;
    this->n = N;
}

Point * Superquadric::applyTransforms(Point * p)
{
    //std::cout << "Translate: " << this->t.unapply(p);
    //std::cout << "Rotate   : " << this->r.unapply(this->t.unapply(p));
    //std::cout << "Scale    : " << this->s.unapply(this->r.unapply(this->t.unapply(p)));
    Point * res = this->s.unapply(this->r.unapply(this->t.unapply(p)));

    return res;
}

// Check if a given point is inside or outside of the 
float Superquadric::isq(Point *p)
{
    Point * transP = this->applyTransforms(p);
    //std::cout << "ISQ --\n" << p;
    //std::cout << transP;
    return pow(pow(transP->X() * transP->X(), 1 / this->e) +
               pow(transP->Y() * transP->Y(), 1 / this->e),
               this->e / this->n) +
           pow(transP->Z() * transP->Z(), 1 / this->n) - 1;
}

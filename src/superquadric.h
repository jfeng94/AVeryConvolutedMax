#ifndef  SUPERQUADRIC_H
#define  SUPERQUADRIC_H

#include "point.h"
#include "matrix.h"

class Superquadric {
    private:
        traMat t;   // Position members
        rotMat r;   // Orientation
        scaMat s;   // Scaling
        float e, n; // Eccentricity values
    
    public:
        Superquadric();
        Superquadric(float, float);
        Superquadric(Point*, Point*, Point*, float, float, float);

        Point * applyTransforms(Point *);
        float   isq(Point *);
        float   isq_prime(Point *, Ray);
        Point * isq_g(Point *);
        Point * getNormal(Point *);
        float   get_initial_guess(Ray);
        float   get_intersection(Ray);
        Point * rayTrace(Ray);
};



#endif

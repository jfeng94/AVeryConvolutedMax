#ifndef  SUPERQUADRIC_H
#define  SUPERQUADRIC_H

#include "point.h"
#include "matrix.h"

class Superquadric {
    private:
        traMat * t;   // Position members
        rotMat * r;   // Orientation
        scaMat * s;   // Scaling
        float e, n; // Eccentricity values
    
    public:
        Superquadric();
        Superquadric(float, float);
        Superquadric(Point*, Point*, Point*, float, float);
        float isq(Point *);
        bool contains(Point *); // Checks inside-outsideness of point 
};



#endif

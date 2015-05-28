#ifndef  SUPERQUADRIC_H
#define  SUPERQUADRIC_H

#include "superquadric.cpp"
#include "geometry.h"


class Superquadric {
    private:
        float x, y, z; // Position members
                       // Orientation
                       // Scaling
        float e, n;    // Eccentricity values
    
    public:
       bool contains(Point *); // Checks inside-outsideness of point 
};



#endif

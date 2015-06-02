#ifndef  SUPERQUADRIC_H
#define  SUPERQUADRIC_H

#include "point.h"
#include "matrix.h"
#include <vector>

class Superquadric {
    private:
        traMat t;   // Position members
        rotMat r;   // Orientation
        scaMat s;   // Scaling
        float e, n; // Eccentricity values

        // Light properties
        Point diffuse;
        Point ambient;
        Point specular;
        float shine;
        float snell;
        float opacity;
    
    public:
        Superquadric();
        Superquadric(float, float);
        Superquadric(Point*, Point*, Point*, float, float, float);
        Superquadric(Point*, Point*, Point*, float, float, float,
                     Point*, Point*, Point*, float, float, float);

        Point * applyTransforms(Point *);
        Point * applyDirTransforms(Point *);
        float   isq(Point *);
        float   isq_prime(Point *, Ray);
        Point * isq_g(Point *);
        Point * getNormal(Point *);
        float   get_initial_guess(Ray);
        float   get_intersection(Ray);
        Point * lighting(Point * p, Point * n, Point * lookFrom, std::vector<pointLight>);
        void    rayTrace(Ray&, Point * lookFrom, std::vector<pointLight>);
};



#endif

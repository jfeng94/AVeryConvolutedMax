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
        // Constructors
        Superquadric();
        Superquadric(float, float);
        Superquadric(Point*, Point*, Point*, float, float, float);
        Superquadric(Point*, Point*, Point*, float, float, float,
                     Point*, Point*, Point*, float, float, float);

        // Point Transformation functions
        Point * applyTransforms(Point *);
        Point * applyDirTransforms(Point *);
        Point * revertTransforms(Point *);
        Point * revertDirTransforms(Point *);

        // Superquadric functions
        float   isq(Point *);
        float   isq_prime(Point *, Ray);
        Point * isq_g(Point *);
        Point * getNormal(Point *);

        // Basic raytracing functions
        float   get_initial_guess(Ray);
        float   get_intersection(Ray);
        void    rayTrace(Ray&, Point * lookFrom, std::vector<pointLight>);

        // Light modeling functions 
        Point * lighting(Point * p, Point * n, Point * lookFrom,
                         std::vector<pointLight>);

        bool    shadow(Point * p, std::vector<pointLight>);
};



#endif

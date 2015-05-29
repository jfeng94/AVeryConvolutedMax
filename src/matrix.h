#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

#include "point.h"

// Base matrix class
class Matrix
{
    protected:
        Point * xyz;

    public:
        virtual Point *   apply(Point *) = 0;
        virtual Point * unapply(Point *) = 0;
};

// Rotation matrix
class rotMat : public Matrix
{
    protected:
        float theta;           // Rotation component

    public:
        rotMat();              // Default constructor
        rotMat(float, float, float, float);
        rotMat(Point*, float); // Normal  constructor

        Point *   apply(Point *);
        Point * unapply(Point *);

};

// Scaling matrix
class scaMat : public Matrix
{
    public:
        scaMat();
        scaMat(float, float, float);
        scaMat(Point*);

        Point *   apply(Point *);
        Point * unapply(Point *);
};

// Translation matrix
class traMat : public Matrix
{
    public:
        traMat();
        traMat(float, float, float);
        traMat(Point *);

        Point *   apply(Point *);
        Point * unapply(Point *);
};
#endif

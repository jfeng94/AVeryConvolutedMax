#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <cstdlib>
#include "point.h"

// Base matrix class
class Matrix
{
    protected:
        Point xyz;

    public:
        void set(Point *);
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

        void setTheta(float t) {this->theta = t;}

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

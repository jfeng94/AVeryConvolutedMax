#ifndef MATRIX_H
#define MATRIX_H

#include "matrix.cpp"
#include "point.h"

// Base matrix class
class Matrix
{
    private:
        float x_comp, y_comp, z_comp;

    public:
        virtual Point *   apply(Point *);
        virtual Point * unapply(Point *);
};

// Rotation matrix
class rotMat : public Matrix
{
    private:
        float theta;           // Rotation component

    public:
        rotMat();              // Default constructor
        rotMat(Point*, float); // Normal  constructor

};

// Scaling matrix
class scaMat : public Matrix
{
    public:
        scaMat();
        scaMat(Point*);
};

// Translation matrix
class traMat : public Matrix
{
    public:
        traMat();
        traMat(Point*);
};
#endif

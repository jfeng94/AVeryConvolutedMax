#ifndef POINT_H
#define POINT_H

#include "point.h"

#endif

#include "isq.cpp"

void set_normal(Point * end, Point * normal_v, float e, float n);

float superquadric(float x, float y, float z, float e, float n);

void superquadric_gradient(Point * p, float e, float n, Point * g);

float superquadric_prime(Point * origin, Point * direction,
                         float t, float e, float n);

float get_intersection(Point * origin, Point * direction, Point * r,
                     float e, float n, float NSCALE);

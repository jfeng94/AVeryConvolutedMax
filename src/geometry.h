#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "geometry.cpp"

class Point {
    private:
        float x, y, z;

    public:
        // Constructors
        Point();
        Point(float, float, float);

        // Accessors
        float X() {return this->x;} 
        float Y() {return this->y;} 
        float Z() {return this->z;} 
        
        // Mutators
        void setX(float X) {this->x = X;}      
        void setY(float Y) {this->y = Y;}
        void setZ(float Z) {this->z = Z;}

        // Other functions
        Point * norm();

        // Operator overloads
        Point * operator+ (Point);
        Point * operator- (Point);
        Point * operator/ (Point);
        Point * operator* (Point);
        Point * operator+=(Point);
        Point * operator-=(Point);
        Point * operator/=(Point);
        Point * operator*=(Point);
        Point * operator= (Point);
        bool    operator==(Point);

        Point * operator+ (float);
        Point * operator- (float);
        Point * operator/ (float);
        Point * operator* (float);
        Point * operator+=(float);
        Point * operator-=(float);
        Point * operator/=(float);
        Point * operator*=(float);
};

class Ray {
    private:
        Point * dir;

    public:
        Ray();
        Ray(int, int, int);
        Ray(Point*);
};

#endif

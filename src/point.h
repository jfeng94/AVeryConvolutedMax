#ifndef POINT_H
#define POINT_H

#include "point.cpp"

class Point 
{
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
        float * dist(Point*);

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
        
        friend std::ostream& operator<< (std::ostream&, Vertex *);
        friend std::ostream& operator<< (std::ostream&, Vertex);
};

// 3D ray class that inherits from point.
class Ray : public Point
{
    private:
        int R, G, B; // Returned color value
        float d;     // Distance to closest object
    public:
        Ray();
        Ray(float, float, float);
        Ray(Point*);

        void setColor(int, int, int);
};

#endif

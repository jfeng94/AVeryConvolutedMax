#ifndef POINT_H
#define POINT_H

#include <iostream>

class Point 
{
    protected:
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
        void set(float X, float Y, float Z) 
        {
            this->x = X;
            this->y = Y;
            this->z = Z;
        }
        void set(Point*p)
        {
            this->x = p->x;
            this->y = p->y;
            this->z = p->z;
        }

        // Other functions
        Point * norm();
        float   dot(Point*);
        Point * cross(Point*);
        float   dist(Point*);
        Point * cwiseMin(Point *);

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
        
        friend std::ostream& operator<< (std::ostream&, Point *);
};

// 3D ray class that inherits from point.
class Ray : public Point
{
    protected:
        float posx, posy, posz;
        int R, G, B; // Returned color value
        float d;     // Distance to closest object
    public:
        Ray();
        Ray(float, float, float, float, float, float);
        Ray(Point*, Point *);

        void setColor(int, int, int);
        void setStart(Point*);
        void setDir(Point*);
        Point * getStart() {return new Point(this->posx, this->posy, this->posz);}
        Point * getDir()   {return new Point(this->x, this->y, this->z);}
        float getR() {return this->R;}
        float getG() {return this->G;}
        float getB() {return this->B;}
        Point * propagate(float);
};

// Point light source in 3D coordinates
class pointLight : public Point
{
    protected:
        int R, G, B;
        float attenuation_k;

    public:
        pointLight();
        pointLight(float, float, float, int, int, int, float);
        pointLight(Point *, int, int, int, float);

        void    setColor(int, int, int);
        Point * getColor();
        void    setAtt_k(float);
        float   getAtt_k();
        void    setPos(Point *p);
        void    setPos(float, float, float);
        Point * getPos();

};
#endif

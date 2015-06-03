#ifndef POINT_H
#define POINT_H

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

class Point 
{
    protected:
        float x, y, z;

    public:
        // Constructors
        __host__ __device__ Point();
        __host__ __device__ Point(float, float, float);

        // Accessors
        __host__ __device__ float X() {return this->x;} 
        __host__ __device__ float Y() {return this->y;} 
        __host__ __device__ float Z() {return this->z;} 
        
        // Mutators
        __host__ __device__ void setX(float X) {this->x = X;}      
        __host__ __device__ void setY(float Y) {this->y = Y;}
        __host__ __device__ void setZ(float Z) {this->z = Z;}
        __host__ __device__ void set(float X, float Y, float Z) 
        {
            this->x = X;
            this->y = Y;
            this->z = Z;
        }
        __host__ __device__ void set(Point*p)
        {
            this->x = p->x;
            this->y = p->y;
            this->z = p->z;
        }

        // Other functions
        __host__ __device__ Point * norm();
        __host__ __device__ float   dot(Point*);
        __host__ __device__ Point * cross(Point*);
        __host__ __device__ float   dist(Point*);
        __host__ __device__ Point * cwiseMin(Point *);

        // Operator overloads
        __host__ __device__ Point * operator+ (Point);
        __host__ __device__ Point * operator- (Point);
        __host__ __device__ Point * operator/ (Point);
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
        float t;     // time to closest object
    public:
        Ray();
        Ray(float, float, float, float, float, float);
        Ray(Point*, Point *);

        // Mutation functions
        __host__ __device__ void setColor(int, int, int);
        __host__ __device__ void setStart(Point*);
        __host__ __device__ void setDir(Point*);
        void setTime(float T) {this->t = T;}

        // Accessor functions
        float getR() {return this->R;}
        float getG() {return this->G;}
        float getB() {return this->B;}
        __host__ __device__ float getTime() {return this->t;}
        __host__ __device__ Point * getStart() {return new Point(this->posx, this->posy, this->posz);}
        __host__ __device__ Point * getDir()   {return new Point(this->x, this->y, this->z);}
        __host__ __device__ Point * propagate(float);
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

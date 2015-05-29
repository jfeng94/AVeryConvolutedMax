#include <float.h>
#include <cmath>
#include "point.h"

// Default constructor
Point::Point()
{
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

// Normal constructor
Point::Point(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

// Returns the norm of a point treated like a vector
Point * Point::norm()
{
    Point * result = new Point();
    *result = *this;

    float mag = sqrt(this->x * this->x +
                     this->y * this->y +
                     this->z * this->z);

    *result /= mag;

    return result;
}

// Calculates the Euclidean distance between 2 points
float Point::dist(Point * p)
{
    float dx = p->x - this->x;
    float dy = p->y - this->y;
    float dz = p->z - this->z;

    return sqrt(dx * dx + dy * dy + dz * dz);
}

// OPERATOR OVERLOADS
Point * Point::operator+(Point p)
{
    Point * result = new Point(this->x + p.x,
                               this->y + p.y,
                               this->z + p.z);
    return result;
}

Point * Point::operator-(Point p)
{
    Point * result = new Point(this->x - p.x,
                               this->y - p.y,
                               this->z - p.z);
    return result;
}

Point * Point::operator*(Point p)
{
    Point * result = new Point(this->x * p.x,
                               this->y * p.y,
                               this->z * p.z);
    return result;
}

Point * Point::operator/(Point p)
{
    Point * result = new Point(this->x / p.x,
                               this->y / p.y,
                               this->z / p.z);
    return result;
}

Point * Point::operator+=(Point p)
{
    this->x += p.x;
    this->y += p.y;
    this->z += p.z;

    return this;
}

Point * Point::operator-=(Point p)
{
    this->x -= p.x;
    this->y -= p.y;
    this->z -= p.z;

    return this;
}

Point * Point::operator*=(Point p)
{
    this->x *= p.x;
    this->y *= p.y;
    this->z *= p.z;

    return this;
}

Point * Point::operator/=(Point p)
{
    this->x /= p.x;
    this->y /= p.y;
    this->z /= p.z;

    return this;
}

Point * Point::operator+(float f)
{
    Point * result = new Point(this->x + f,
                               this->y + f,
                               this->z + f);
    return result;
}

Point * Point::operator-(float f)
{
    Point * result = new Point(this->x - f,
                               this->y - f,
                               this->z - f);
    return result;
}

Point * Point::operator*(float f)
{
    Point * result = new Point(this->x * f,
                               this->y * f,
                               this->z * f);
    return result;
}

Point * Point::operator/(float f)
{
    Point * result = new Point(this->x / f,
                               this->y / f,
                               this->z / f);
    return result;
}

Point * Point::operator+=(float f)
{
    this->x += f;
    this->y += f;
    this->z += f;

    return this;
}

Point * Point::operator-=(float f)
{
    this->x -= f;
    this->y -= f;
    this->z -= f;
    return this;
}

Point * Point::operator*=(float f)
{
    this->x *= f;
    this->y *= f;
    this->z *= f;
    return this;
}

Point * Point::operator/=(float f)
{
    this->x /= f;
    this->y /= f;
    this->z /= f;
    return this;
}

Point * Point::operator=(Point p)
{
    Point * result = new Point(p.x, p.y, p.z);
    return result;
}

bool Point::operator==(Point p)
{
    if (this->x == p.x && this->y == p.y && this->z == p.z)
    {
        return true;
    }

    return false;
}


// Stream output operator
std::ostream& operator<<(std::ostream &out, Point *p)
{
    out << p->x << "\t" << p->y << "\t" << p->z << "\n";
    return out;
}

/******************************************************************************/
/*                           Methods for Ray Class                            */
/******************************************************************************/
Ray::Ray()
{
    // No direction
    this->x = 0;
    this->y = 0;
    this->z = 0;
    
    // Default color black
    this->R = 0;
    this->G = 0;
    this->B = 0;
    this->d = FLT_MAX;
}

Ray::Ray(float X, float Y, float Z)
{
    this->x = X;
    this->y = Y;
    this->z = Z;

    // Default color black
    this->R = 0;
    this->G = 0;
    this->B = 0;
    this->d = FLT_MAX;
}

Ray::Ray(Point* p)
{
    this->x = p->X();
    this->y = p->Y();
    this->z = p->Z();
    
    // Default color black./
    this->R = 0;
    this->G = 0;
    this->B = 0;
    this->d = FLT_MAX;
}

void Ray::setColor(int r, int g, int b)
{
    this->R = r;
    this->G = g;
    this->B = b;
}

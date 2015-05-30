#include <cmath>
#include <iostream>

using namespace std;

/* Simple point structure to be used with ray tracing */
struct Point
{
    float x;
    float y;
    float z;
};

// Copy function
inline void copy(Point * a, Point *b)
{
    b->x = a->x;
    b->y = a->y;
    b->z = a->z;
}

// Normalization function
inline void normalize(Point * a)
{
    float norm = sqrt(a->x * a->x + a->y * a->y + a->z * a->z);
    a->x /= norm;
    a->y /= norm;
    a->z /= norm;

}

// Get distance between two points
inline float distance(Point * a, Point *b )
{
    return sqrt((a->x - b->x) * (a->x - b->x) +
                (a->y - b->y) * (a->y - b->y) + 
                (a->z - b->z) * (a->z - b->z));
}

// Simple function to calculate ray properties.
inline void ray(Point * origin, Point * direction, Point * r, float time)
{
    r->x = origin->x + time * direction->x;
    r->y = origin->y + time * direction->z;
    r->z = origin->z + time * direction->y;
}

// Function that untransforms a ray.
inline void untransform(Point * o, Point * d,
                        float xt, float yt, float zt,
                        float rotation, float rotx, float roty, float rotz,
                        float length, float width, float depth)
{
    float x, y, z, dx, dy, dz;
    x = o->x;
    y = o->y;
    z = o->z;

    dx = d->x;
    dy = d->y;
    dz = d->z;

    // First, untranslate start point (unnecessary for direction vector)
    x -= xt;
    y -= yt;
    z -= zt;

    // Prepare unrotation matrix
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;
    float r1, r2, r3, theta;
    float rx, ry, rz, drx, dry, drz;

    // unRotation vector must be unit.
    float norm = sqrt(rotx * rotx + roty * roty + rotz * rotz);
    r1 = rotx / norm;
    r2 = roty / norm;
    r3 = rotz / norm;
    theta = (360 - rotation) * M_PI / 180;

    m11 = r1 * r1 + cos(theta) * (1 - r1 * r1);
    m21 = r2 * r1 - cos(theta) * r2 * r1 + r3 * sin(theta);
    m31 = r2 * r1 - cos(theta) * r3 * r1 - r2 * sin(theta);

    m12 = r1 * r2 - cos(theta) * r1 * r2 - r3 * sin(theta);
    m22 = r2 * r2 + cos(theta) * (1 - r2 * r2);
    m32 = r3 * r2 - cos(theta) * r3 * r2 + r1 * sin(theta);

    m13 = r1 * r3 - cos(theta) * r1 * r3 + r2 * sin(theta);
    m23 = r2 * r3 - cos(theta) * r2 * r3 - r1 * sin(theta);
    m33 = r3 * r3 + cos(theta) * (1 - r3 * r3);

    // Do unrotations
    rx = m11 * x + m12 * y + m13 * z;
    ry = m21 * x + m22 * y + m23 * z;
    rz = m31 * x + m32 * y + m33 * z;

    drx = m11 * dx + m12 * dy + m13 * dx;
    dry = m21 * dx + m22 * dy + m23 * dz;
    drz = m31 * dx + m32 * dy + m33 * dz;

    x = rx;
    y = ry;
    z = rz;

    dx = drx;
    dy = dry;
    dz = drz;

    // Unscale these guys
    x /= length;
    y /= width;
    z /= depth;

    dx /= length;
    dy /= width;
    dz /= depth;
    
    // Save changes
    o->x = x;
    o->y = y;
    o->z = z;

    d->x = dx;
    d->y = dy;
    d->z = dz;
}

// Function that retransforms a ray.
inline void transform(Point * o, Point * d,
                      float xt, float yt, float zt,
                      float rotation, float rotx, float roty, float rotz,
                      float length, float width, float depth)
{
    float x, y, z, dx, dy, dz;
    x = o->x;
    y = o->y;
    z = o->z;

    dx = d->x;
    dy = d->y;
    dz = d->z;

    // Unscale these guys
    x *= length;
    y *= width;
    z *= depth;

    dx *= length;
    dy *= width;
    dz *= depth;

    // Prepare rotation matrix
    float m11, m12, m13, m21, m22, m23, m31, m32, m33;
    float r1, r2, r3, theta;
    float rx, ry, rz, drx, dry, drz;

    // Rotation vector must be unit.
    float norm = sqrt(rotx * rotx + roty * roty + rotz * rotz);
    r1 = rotx / norm;
    r2 = roty / norm;
    r3 = rotz / norm;
    theta = rotation * M_PI / 180;

    m11 = r1 * r1 + cos(theta) * (1 - r1 * r1);
    m21 = r2 * r1 - cos(theta) * r2 * r1 + r3 * sin(theta);
    m31 = r2 * r1 - cos(theta) * r3 * r1 - r2 * sin(theta);

    m12 = r1 * r2 - cos(theta) * r1 * r2 - r3 * sin(theta);
    m22 = r2 * r2 + cos(theta) * (1 - r2 * r2);
    m32 = r3 * r2 - cos(theta) * r3 * r2 + r1 * sin(theta);

    m13 = r1 * r3 - cos(theta) * r1 * r3 + r2 * sin(theta);
    m23 = r2 * r3 - cos(theta) * r2 * r3 - r1 * sin(theta);
    m33 = r3 * r3 + cos(theta) * (1 - r3 * r3);

    // Do rotations
    rx = m11 * x + m12 * y + m13 * z;
    ry = m21 * x + m22 * y + m23 * z;
    rz = m31 * x + m32 * y + m33 * z;

    drx = m11 * dx + m12 * dy + m13 * dx;
    dry = m21 * dx + m22 * dy + m23 * dz;
    drz = m31 * dx + m32 * dy + m33 * dz;

    x = rx;
    y = ry;
    z = rz;

    dx = drx;
    dy = dry;
    dz = drz;

    // Lastly, retranslate start point (unnecessary for direction vector)
    x += xt;
    y += yt;
    z += zt;

    
    // Save changes
    o->x = x;
    o->y = y;
    o->z = z;

    d->x = dx;
    d->y = dy;
    d->z = dz;
}

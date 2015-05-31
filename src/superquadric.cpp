#include "superquadric.h"
#include "matrix.h"
#include "point.h"

#include <math.h>
#include <iostream>

// Create a sphere at the origin of the scene.
Superquadric::Superquadric()
{
    this->e = 1;
    this->n = 1;
}


// Create an object with specified eccentricities at the origin.
Superquadric::Superquadric(float E, float N)
{
    this->e = E;
    this->n = N;
}

// Fully customized superquadric constructor.
Superquadric::Superquadric(Point * tra, Point * sca, Point * rot,
                           float theta, float E, float N)
    {
    this->t.set(tra);
    this->s.set(sca);
    this->r.set(rot->norm());
    this->r.setTheta(theta);
    this->e = E;
    this->n = N;
}

Point * Superquadric::applyTransforms(Point * p)
{
    //std::cout << "Translate: " << this->t.unapply(p);
    //std::cout << "Rotate   : " << this->r.unapply(this->t.unapply(p));
    //std::cout << "Scale    : " << this->s.unapply(this->r.unapply(this->t.unapply(p)));
    Point * res = this->s.unapply(this->r.unapply(this->t.unapply(p)));

    return res;
}

// Check if a given point is inside or outside of the 
float Superquadric::isq(Point *p)
{
    Point * transP = this->applyTransforms(p);
    //std::cout << "ISQ --\n" << p;
    //std::cout << transP;
    return pow(pow(transP->X() * transP->X(), 1 / this->e) +
               pow(transP->Y() * transP->Y(), 1 / this->e),
               this->e / this->n) +
           pow(transP->Z() * transP->Z(), 1 / this->n) - 1;
}

// Get the gradient vector of the superquadric
Point * Superquadric::isq_g(Point * p)
{
    float x, y, z, gx, gy, gz;
    x = p->X();
    y = p->Y();
    z = p->Z();

    gx = 2 * x * pow((x * x), (1.0 / this->e - 1)) *
         pow(pow((x * x), (1.0 / this->e)) + pow((y * y), (1.0 / this->e)), (this->e / this->n - 1)) / this->n;
    gy = 2 * y * pow((y * y), (1.0 / this->e - 1)) *
         pow(pow((x * x), (1.0 / this->e)) + pow((y * y), (1.0 / this->e)), (this->e / this->n - 1)) / this->n;
    gz = 2 * z * pow((z * z), (1.0 / this->n - 1)) / this->n;

    return new Point(gx, gy, gz);
}

// Get the derivative at a point emanating from a ray.
float Superquadric::isq_prime(Point *p, Ray r)
{
    Point * g = this->isq_g(p);;
    
    return g->dot(r.getDir());
}

// Function to get initial time guess
float Superquadric::get_initial_guess(Ray r)
{
    float a, b, c, discriminant;
    a = (r.getDir())->dot(r.getDir());
    b = (r.getStart())->dot(r.getDir());
    c = (r.getStart())->dot(r.getStart());

    if (b < 0)
    {
        a *= -1;
        b *= -1;
        c *= -1;
    }

    discriminant = b * b - 4 * a * c;

    // If discriminant is less than 0. AKA, misses object
    if (discriminant < 0)
        return 0;

    float t_plus, t_minus;
    t_minus = (-1.0 * b - sqrt(discriminant)) / (2 * a);
    t_plus  = (2 * c) / (-1.0 * b - sqrt(discriminant));

    return (t_minus < t_plus) ? t_minus : t_plus;
}


// Function takes a ray from the camera, and finds an intersection point
// with the object. If an intersection is found, the color field of the 
// ray is updated.
float Superquadric::get_intersection(Ray r)
{
    float t_old, t_new;
    bool done;
    float g, g_prime;

    // Get the time to propagate to the bounding sphere
    t_old = this->get_initial_guess(r);
    t_new = t_old;

    if (t_old == 0)
    {
        return 0;
    }
    
    // Propagate the ray to the bounding sphere
    Point * intersect = r.propagate(t_old);
    
    done = false;
    int iterations = 0;
    while(iterations < 100)
    {
        iterations++;

        // Update t_old
        t_old = t_new;
        g       = this->isq(intersect);
        g_prime = this->isq_prime(intersect, r);

        // g'(x) = 0 and g ~ 0. Don't update time!
        if (g < 0.001 && g_prime > -0.001)
        {
            done = true;
            return t_old;
        }
        // g'(x) = 0 but g not close to 0. No intersection.
        else if (g_prime == 0)
        {
            return 0;
        }
        // g'(x) changes sign. We've exited the bounding sphere. No intersect.
        else if (g_prime > 0)
        {
            return 0;
        }
        else if (g <= 1e-4)
        {
            done = true;
            return t_old;
        }

        // Update new time guess
        t_new = t_old * (g / g_prime);

        // Find the new intersection point.
        intersect = r.propagate(t_new);
    }

    // Return found time
    return t_new;
}

Point * Superquadric::getNormal(Point * p)
{
    float norm, x, y, z;

    Point * n = this->isq_g(p);
    n = n->norm();
    
    return n;
}

void rayTrace(Ray r)
{
    
}

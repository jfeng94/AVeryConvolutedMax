#include "superquadric.h"
#include "matrix.h"
#include "point.h"

#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <float.h>

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

    this->ambient = *(new Point(255, 255, 255));
    this->specular = *(new Point(155, 155, 155));
    this->diffuse = *(new Point(155, 155, 155));
}

// Fully customized superquadric constructor.
Superquadric::Superquadric(Point * tra, Point * sca, Point * rot,
                           float theta, float E, float N,
                           Point * dif, Point * amb, Point * spe,
                           float shi, float sne, float opa )
{
    this->t.set(tra);
    this->s.set(sca);
    this->r.set(rot->norm());
    this->r.setTheta(theta);
    this->e = E;
    this->n = N;

    this->diffuse  = *dif;
    this->ambient  = *amb;
    this->specular = *spe;

    this->shine   = shi;
    this->snell   = sne;
    this->opacity = opa;
}

Point * Superquadric::applyTransforms(Point * p)
{
    //std::cout << "Translate: " << this->t.unapply(p);
    //std::cout << "Rotate   : " << this->r.unapply(this->t.unapply(p));
    //std::cout << "Scale    : " << this->s.unapply(this->r.unapply(this->t.unapply(p)));
    Point * res = this->s.unapply(this->r.unapply(this->t.unapply(p)));

    return res;
}

Point * Superquadric::applyDirTransforms(Point *p)
{
    return this->s.unapply(this->r.unapply(p));
}

// Check if a given point is inside or outside of the 
float Superquadric::isq(Point *p)
{
    Point * transP = p;
    //Point * transP = this->applyTransforms(p);
    //std::cout << "ISQ --\n" << p;
    //std::cout << transP;
    return pow(pow(transP->X() * transP->X(), (double) 1 / e) +
               pow(transP->Y() * transP->Y(), (double) 1 / e),
               e / n) +
           pow(transP->Z() * transP->Z(), (double) 1 / n) - 1;
}

// Get the gradient vector of the superquadric
Point * Superquadric::isq_g(Point * p)
{
    float x, y, z, gx, gy, gz;
    x = p->X();
    y = p->Y();
    z = p->Z();
    //e = this->e;
    //n = this->n;

    if (n == 0)
    {
        //std::cout << "n is 0!\n";
        gx = gy = gz = FLT_MAX;
    }
    else if (e == 0)
    {
        //std::cout << "e is 0!\n";
        gx = gy = FLT_MAX;
        gz = (2 * z * pow(pow(z, 2), ((double) 1 / n) - 1)) / (double) n;
    }
    else
    {
        float xterm = pow(pow(x, 2), (double) 1 / e);
        float yterm = pow(pow(y, 2), (double) 1 / e);
        float xyterm = pow(xterm + yterm, ((double) e /n ) - 1);
        float x2term = (2 * x * pow(pow(x, 2), ((double) 1 / e) - 1));
        float y2term = (2 * y * pow(pow(y, 2), ((double) 1 / e) - 1));
        gx = x2term * xyterm / (double) n;
        gy = y2term * xyterm / (double) n;
        gz = (2 * z * pow(pow(z, 2), ((double) 1 / n) - 1)) / (double) n;
    }
    //gx = 2 * x * pow((x * x), (1.0 / this->e - 1)) *
    //     pow(pow((x * x), (1.0 / this->e)) + pow((y * y), (1.0 / this->e)), (this->e / this->n - 1)) / this->n;
    //gy = 2 * y * pow((y * y), (1.0 / this->e - 1)) *
    //     pow(pow((x * x), (1.0 / this->e)) + pow((y * y), (1.0 / this->e)), (this->e / this->n - 1)) / this->n;
    //gz = 2 * z * pow((z * z), (1.0 / this->n - 1)) / this->n;

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
    b = 2 * (r.getStart())->dot(r.getDir());
    c = (r.getStart())->dot(r.getStart()) - 3;

    if (b < 0)
    {
        a *= -1;
        b *= -1;
        c *= -1;
    }

    discriminant = b * b - 4 * a * c;
    //std::cout << "a: " << a << "\tb: " << b << "\tc: " << c  << "\tdiscriminant: " << discriminant << "\n";
    

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

    std::ofstream out;
    out.open("MatlabTools/TestRay.txt", std::fstream::app);
    
    // Get the time to propagate to the bounding sphere
    t_old = this->get_initial_guess(r);
    //std::cout << t_old << "\n";
    t_new = t_old;

    if (t_old == 0)
    {
        //std::cout << "T_old = 0\n";
        return 0;
    }
    
    // Propagate the ray to the bounding sphere
    Point * intersect = r.propagate(t_old);
    
    done = false;
    int iterations = 0;
    while(iterations < 100)
    {
        iterations++;

        out << intersect;
        // Update t_old
        t_old = t_new;
        g       = this->isq(intersect);
        g_prime = this->isq_prime(intersect, r);
        //std::cout << "Time: " << t_new << "\n";
        //std::cout << "G: " << g << " G': " << g_prime << "\n";

        // g'(x) = 0 and g ~ 0. Don't update time!
        if (g < 0.001 && g_prime > -0.001)
        {
            //std::cout << "Case 1\n";
            done = true;
            out << intersect;
            return t_old;
        }
        // g'(x) = 0 but g not close to 0. No intersection.
        else if (g_prime == 0)
        {
            //std::cout << "Case 2\n";
            return 0;
        }
        // g'(x) changes sign. We've exited the bounding sphere. No intersect.
        else if (g_prime > 0)
        {
            //std::cout << "Case 3\n";
            return 0;
        }
        else if (g <= 1e-4)
        {
            //std::cout << "Case 4\n";
            done = true;
            out << intersect;
            return t_old;
        }

        // Update new time guess
        t_new = t_old - (g / g_prime);

        // Find the new intersection point.
        intersect = r.propagate(t_new);
    }

    //std::cout << "Case 5\n";
    // Return found time
    out << intersect;
    return t_new;
}

Point * Superquadric::getNormal(Point * p)
{
    float norm, x, y, z;

    Point * n = this->isq_g(p);
    n = n->norm();
    
    return n;
}

// Lighting contribution algorithm: Phong's
Point * Superquadric::lighting(Point * p, Point * n, Point * lookFrom,
                               std::vector<pointLight> lights)
{
    Point * difSum = new Point(0, 0, 0);
    Point * speSum = new Point(0, 0, 0);
    Point * reflLight = new Point(0, 0, 0);
    Point * refrLight = new Point(0, 0, 0);
    
    // Direction from point to source (Usually camera)
    Point * eDir = (*lookFrom - *p)->norm();

    for (int i = 0; i < lights.size(); i++)
    {
        // Solve for attenuation term
        Point * lP  = lights[i].getPos();
        Point * lC  = lights[i].getColor();
        float att_k = lights[i].getAtt_k();

        std::cout << "Applying Light: " << lP << lC << "To Point: " << p << "\n\n";

        Point * lDir = (*lP - *p)->norm();
        float lightDist = lP->dist(p); 

        float atten = 1 / (float) (1 + (att_k * lightDist * lightDist));

        // Add diffusion factor to sum
        float nDotl = n->dot(lDir);
        Point * lDif = *lC * (atten * ((0 < nDotl) ? nDotl : 0));
        *difSum += *lDif;

        // Add specular factor to sum
        Point *dirDif = (*eDir + *lDir)->norm();
        float nDotDir = n->dot(dirDif);
        Point * lSpe  = *lC * (atten * pow(((0 < nDotDir && 0 < nDotl) ? nDotDir : 0), shine));
        *speSum += *lSpe;
    }
    
    Point * min = new Point(255, 255, 255);
    Point * result =  min->cwiseMin(*(*(*difSum / 255) * this->diffuse) +
                                    *(*(*speSum / 255) * this->specular));
    //std::cout << "Normal v:  " << n;
    //std::cout << "Diffusion: " << *difSum * this->diffuse;
    //std::cout << "Specular:  " << *speSum * this->specular;
    //std::cout << result;
    return result;
} 

void Superquadric::rayTrace(Ray &r, Point * lookFrom, std::vector<pointLight> lights)
{
    Point * origin = r.getStart();
    Point * dir    = r.getDir();

    // Transform frame of reference so that this object is at origin.
    origin = this->applyTransforms(origin);
    dir    = (this->applyDirTransforms(dir))->norm();
    
    //std::cout << "Transformed origin: " << origin;
    //std::cout << "Transformed direct: " << dir;

    // Create new ray to do intersection test
    Ray transR;
    transR.setStart(origin);
    transR.setDir(dir);

    // Check for intersection
    float intersects = get_intersection(transR);
    
    std::ofstream out;
    out.open("MatlabTools/TestNormals.txt", std::fstream::app);

    if (intersects != 0)
    {
        // Calculate the intersection point
        Point * pTran = transR.propagate(intersects);
        Point * pTrue = r.propagate(intersects);


        // Get the normal at the intersection point
        Point * n = (this->getNormal(pTran))->norm();

        Point *showNorm = *pTran + *(*n / 10);

        out << pTran->X() << " " << pTran->Y() << " " << pTran->Z() << " " << showNorm;

        Point * color = lighting(pTrue, n, lookFrom, lights);
        //std::cout << "Setting white pixel...\n";
        r.setColor(color->X(), color->Y(), color->Z());
    }
}

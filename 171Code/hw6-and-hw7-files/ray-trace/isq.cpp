#ifndef POINT_H
#define POINT_H

#include "point.h"

#endif

// Tells us whether a point is inside of the object. 
float superquadric(Point * origin, Point * direction, float t, float e, float n)
{
    Point * p = new Point;
    ray(origin, direction, p, t);
    float x, y, z;

    x = p->x;
    y = p->y;
    z = p->z;

    float tempx = pow(x * x, 1/e);
    float tempy = pow(y * y, 1/e);
    float temp  = pow((tempx + tempy), e/n);
    float tempz = pow(z * z, 1/n);
    float val = temp + tempz;

    //cerr << "(" << tempx << " + " << tempy << ") ^ " << e << " / " << n << " = " << temp << "\n"
    //     << temp << " + " << tempz  << " = " << val << "\n";

    free(p);

    return val - 1;
}

void superquadric_gradient(Point * p, float e, float n, Point * g)
{
    float x, y, z, rx, ry, rz;
    x = p->x;
    y = p->y;
    z = p->z;

    //if(verbose)
    //{
    //    cerr << "\t\tg->x: " <<  2 * x << " * "
    //         << pow((x * x), (1.0 / e - 1)) << " * "
    //         << pow(pow((x * x), (1.0 / e)) + pow((y * y), (1.0 / e)), (e / n - 1)) << " / "
    //         << n << "\n"
    //         << "\t\tg->y: " <<  2 * y << " * "
    //         << pow((y * y), (1.0 / e - 1)) << " * "
    //         << pow(pow((x * x), (1.0 / e)) + pow((y * y), (1.0 / e)), (e / n - 1)) << " / "
    //         << n << "\n"
    //         << "\t\tg->z: " << 2 * z << " * "
    //         << pow((z * z), (1.0 / n - 1)) << " / " << n << "\n";
    //}

    g->x = 2 * x * pow((x * x), (1.0 / e - 1)) *
           pow(pow((x * x), (1.0 / e)) + pow((y * y), (1.0 / e)), (e / n - 1)) / n;
    g->y = 2 * y * pow((y * y), (1.0 / e - 1)) *
           pow(pow((x * x), (1.0 / e)) + pow((y * y), (1.0 / e)), (e / n - 1)) / n;
    g->z = 2 * z * pow((z * z), (1.0 / n - 1)) / n;
}

float superquadric_prime(Point * origin, Point * direction,
                         float t, float e, float n)
{
    Point * x = new Point;
    Point * g = new Point;
    //if(verbose)
    //{
    //    cerr << "GETTING RAY:\n";
    //    cerr << "\tPassing origin: (" << origin->x << ", " << origin->y << ", " << origin->z << ")\n"
    //         << "\t     direction: (" << direction->x << ", " << direction->y << ", " << direction->z << ")\n"
    //         << "\t          time: " << t << "\n";
    //}
    ray(origin, direction, x, t);
    //if(verbose)
    //{
    //    cerr << "\t        Result: (" << x->x << ", " << x->y << ", " << x->z << ")\n" 
    //         << "GETTING GRADIENT:\n"
    //         << "\tPassing point: (" << x->x << ", " << x->y << ", " << x->z << ")\n"
    //         << "\t            e: " << e << "\n"
    //         << "\t            n: " << n << "\n";
    //}
    superquadric_gradient(x, e, n, g);
    //if(verbose)
    //{
    //    cerr << "\tx: " << x->x << "\t" << x->y << "\t" << x->z << "\n";
    //    cerr << "\tg: " << g->x << "\t" << g->y << "\t" << g->z << "\n";
    //    cerr << "\ta: " << direction->x << "\t" << direction->y << "\t" << direction->z << "\n";
    //}

    return direction->x * g->x + direction->y * g->y + direction->z * g->z;
}

float get_initial_guess(Point * origin, Point * direction)
{
    //if(verbose)
    //{
    //    cerr << "\tGiven origin: (" << origin->x << ", " << origin->y << ", " << origin->z << ")\n"
    //         << "\tGiven vector: (" << direction->x << ", " << direction->y << ", " << direction->z << ")\n";
    //}

    float a, b, c, discriminant;
    a = direction->x * direction->x + 
        direction->y * direction->y + 
        direction->z * direction->z;
    b = 2 * (origin->x * direction->x +
             origin->y * direction->y + 
             origin->z * direction->z);
    c = origin->x * origin->x +
        origin->y * origin->y + 
        origin->z * origin->z - 3;
    //if(verbose)
    //{
    //    cerr << "\tCalculating a:\n"
    //         << "\t\t" << direction->x << " * " << direction->x << " + " << direction->y << " * " << direction->y << " + " << direction->z << " * " << direction->z << " = " << a << "\n"
    //         << "\tCalculating b:\n"
    //         << "\t\t2 * (" << origin->x << " * " << direction->x << " + " << origin->y << " * " << direction->y << " + " << origin->z << " * " << direction->z << ") = " << b << "\n"
    //         << "\tCalculating c:\n"
    //         << "\t\t" << origin->x << " * " << origin->x << " + " << origin->y << " * " << origin->y << " + " << origin->z << " * " << origin->z << " - 3 = " << c << "\n";
    //}
    if(b < 0)
    {
        a *= -1;
        b *= -1;
        c *= -1;
    }

    //if(verbose)
    //    cerr << "\tDiscriminant: " << b << " * " << b << " - 4 * " << a << " * " << c << "\n";
    discriminant = b * b - 4 * a * c;

    // Check for complex case
    if(discriminant < 0)
    {
        return 0;
    }

    // Get time candidates.
    float t_plus, t_minus;
    t_minus = (-1.0 * b - sqrt(b * b - 4 * a * c)) / (2 * a);
    t_plus  = (2 * c) / (-1.0 * b - sqrt(b * b - 4 * a * c));

    //if(verbose)
    //    cerr << "\tt_minus: " << t_minus << "\tt_plus: " << t_plus << "\n"; 
    return (t_minus < t_plus) ? t_minus : t_plus;
}

float get_intersection(Point * origin, Point * direction, Point * r, float e, float n, float NSCALE)
{
    bool verbose = false;

    Point * tempo = new Point;
    Point * tempd = new Point;

    copy(origin, tempo);
    copy(direction, tempd);

    float t_old, t_new;
    bool done;
    float g, g_prime;

    // Get initial guess
    t_old = get_initial_guess(tempo, direction);
    t_new = t_old;

    if(t_old == 0)
    {
        // Discriminant less than 0. Complex solutions miss the sphere
        if(verbose)
            cerr << "Discriminant = 0 \n";
        return 0;
    }
    if(verbose)
        cerr << "\n\n\nGiven point: (" << origin->x << ", " << origin->y << ", " << origin->z << ")\n"
             << "In direction: (" << direction->x << ", " << direction->y << ", " << direction->z << ")\n"
             << "=== INITIAL GUESS ===\n"
             << "Time: " << t_new << "\n";
    ray(tempo, direction, r, t_old);

    tempo->y = origin->z;
    tempo->z = origin->y;

    if(verbose)
        cerr << "END: " << r->x << "\t" << r->y << "\t" << r->z << "\n";

    // Start iterations
    done = false;
    int iterations = 0;
    while(iterations < 100)
    {
        iterations++;

        if(verbose)
            cerr << "====== ITERATION " << iterations << " ======\n";

        t_old = t_new;
        g = superquadric(tempo, direction, t_old, e, n);
        g_prime = superquadric_prime(tempo, direction, t_old, e, n);
        
        if(verbose)
            cerr << "g: " << g << " g': " << g_prime << "\n";
        // g'(x) = 0, g ~ 0 case. Don't update t.
        if(g < 0.001 && g_prime > -0.001)
        {
            done = true;
            return t_old;
        }
        // g_prime is 0 but g not close to 0. Stop
        else if(g_prime == 0)
        {
            if(verbose)
            cerr << "No intersection! Iteration terminated at step " << iterations << "\n"
                 << "\t-- e: " << e << " n: " << n << "\n"
                 << "\tg' is 0, but g not close to 0.\n"
                 << "\tg(x) = " << g << "\tg'(x) = " << g_prime << "\n";
            return 0;
       }
        // g'(x) changes sign: general stop point for miss.
        else if(g_prime > 0)
        {
            if(verbose)
            cerr << "No intersection! Iteration terminated at step " << iterations << "\n"
                 << "\t-- e: " << e << " n: " << n << "\n"
                 << "\tg' changes sign.\n"
                 << "\tg(x) = " << g << "\tg'(x) = " << g_prime << "\n";
            return 0;
        }
        // we touch the surface (or are in it) general stop point for hit.
        else if(g <= 0.0001)
        {
            done = true;
            return t_old;
        }
        
        // Update t
        t_new = t_old - NSCALE * (g / g_prime); 

        // Update ray.
        ray(tempo, direction, r, t_new);
        if(verbose)
            cerr << "END: " << r->x << "\t" << r->y << "\t" << r->z << "\n"
                 << "Time: " << t_new << "\n";
    }
    if(verbose)
        cerr << "t_final: " << t_new << "\n";
    return t_new;
}

inline void set_normal(Point * end, Point * normal_v, float e, float n)
{
    float norm, x, y, z;
    
    Point * p = new Point;
    copy(end, p);

    p->y = end->z;
    p->z = end->y;

    superquadric_gradient(end, e, n, normal_v);
    x = normal_v->x;
    y = normal_v->y;
    z = normal_v->z;
    norm = sqrt(x * x + y * y + z * z);
    normal_v->x /= norm;
    normal_v->y /= norm;
    normal_v->z /= norm;
}


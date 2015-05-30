bool HW6 = false;
bool HW7 = !HW6;
#include <Eigen/Eigen>

#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#ifndef POINT_H
#define POINT_H

#include "point.h"

#endif

#ifndef ISQ_H
#define ISQ_H

#include "isq.h"

#endif

float NSCALE = 0.01;

using namespace std;
using namespace Eigen;

// Define camera model variables
Point * LookAt = new Point;
Point * LookFrom = new Point;
Point * Up = new Point;


// Resolution terms
int Nx = 400;
int Ny = 400;

// Film width -- 35mm camera
float Fx = 0.035;
float Fy;

// Focal Length -- distance from film
float Fd = 0.050;

// Struct that stores light information
struct Point_Light
{
    float position[3];
    float color[3];
    float attenuation_k;
};

// Light vector for global access
vector<Point_Light> lights;


void add_light(float x, float y, float z, float R, float G, float B, float k)
{
    Point_Light l;
    l.position[0]   = x;
    l.position[1]   = y;
    l.position[2]   = z;
    l.color[0]      = R;
    l.color[1]      = G;
    l.color[2]      = B;
    l.attenuation_k = k;

    lights.push_back(l);
}

// Function prototypes





// Struct that represents a single object in the space
class Object
{
    // Shape properties
    float e, n;             // Eccentricity variables
    float l, w, d;          // Scaling transform
    float rot, rx, ry, rz;  // Rotation transform
    float xt, yt, zt;       // Translation transform

    // Material properties
    float ca0, ca1, ca2;          // Ambient  reflect
    float cd0, cd1, cd2;          // Diffuse  reflect
    float cs0, cs1, cs2;          // Specular reflect
    float p;                      // Shininess

    int objnum;

    Vector3d Lighting(Point * P, Point * nv, vector<Object *> objects, Point * o);

public:
    // Contructor
    Object(float e, float n,
           float l, float w, float d,
           float r0, float r1, float r2, float r3,
           float t1, float t2, float t3,
           float ca0, float ca1, float ca2,
           float cd0, float cd1, float cd2,
           float cs0, float cs1, float cs2,
           float p, int num) 
           : e(e), n(n), l(l), w(w), d(d),
             rot(r0), rx(r1), ry(r2), rz(r3),
             xt(t1), yt(t2), zt(t3),
             ca0(ca0), ca1(ca1), ca2(ca2),
             cd0(cd0), cd1(cd1), cd2(cd2),
             cs0(cs0), cs1(cs1), cs2(cs2),
             p(p), objnum(num) {}
    void print_obj();
    Vector3d ray_trace(Point * origin, Point * direction, Point * r, vector<Object*> objects);
    bool check_shadow(Point * p , Point * s, float x, float y, float z);
};

void Object::print_obj()
{
    cerr << "e:   " << this->e << "\t"
         << "n:   " << this->n << "\n"
         << "l:   " << this->l << "\t"
         << "w:   " << this->w << "\t"
         << "d:   " << this->d << "\n"
         << "rot: " << this->rot << "\t"
         << "rx:  " << this->rx << "\t"
         << "ry:  " << this->ry << "\t"
         << "rz:  " << this->rz << "\n"
         << "xt:  " << this->xt << "\t"
         << "yt:  " << this->yt << "\t"
         << "zt:  " << this->zt << "\n"
         << "ca:  " << this->ca0 << "\t" << this->ca1 << "\t" << this->ca2 << "\n"
         << "cd:  " << this->cd0 << "\t" << this->cd1 << "\t" << this->cd2 << "\n"
         << "cs:  " << this->cs0 << "\t" << this->cs1 << "\t" << this->cs2 << "\n"
         << "p:   " << this->p << "\n";
}

bool Object::check_shadow(Point * p, Point * s, float x, float y, float z)
{
    // Trash point;
    Point * tempp = new Point;
    Point * temps = new Point;
    Point * end = new Point;
    Point * trash = new Point;

    copy(p, tempp);
    copy(s, temps);

    // Untransform the temp points
    untransform(tempp, temps, 
                this->xt, this->yt, this->zt,
                this->rot, this->rx, this->ry, this->rz,
                this->l, this->w, this->d);

    //tempp->y = p->z;
    //tempp->z = p->y;

    normalize(temps);

    // Get intersection
    float intersects = get_intersection(tempp, s, end, this->e, this->n, NSCALE);
    
    //untransform(end, trash,
    //            this->xt, this->yt, this->zt,
    //            this->rot, this->rx, this->ry, this->rz,
    //            this->l, this->w, this->d);

    float dist1 = sqrt((tempp->x - x) * (tempp->x - x) + (tempp->y - y) * (tempp->y - y) + (tempp->z - z) * (tempp->z - z));
    float dist2 = sqrt((end->x - x) * (end->x - x) + (end->y - y) * (end->y - y) + (end->z - z) * (end->z - z));


    if((intersects != 0) && (dist2 < dist1))
    {
        //cerr << "Point (" << p->x << ", " << p->y << ", " << p->z << ") to (" << x << ", " << y << ", " << z << ") "
        //     << "blocked by point (" << end->x << ", " << end->y << ", " << end->z << "). Time: " << intersects << "\n";
        return true;
    }
    else
        return false;
}

// Function to find color of light for a pixel
Vector3d Object::Lighting(Point * P, Point * nv, vector<Object *> objects, Point * o)
{
    float dist = distance(P, o);
    dist *= dist;


    Point * shadow = new Point;
    Point * tempLook = new Point;
    
    Vector3d dif_sum;
    Vector3d spe_sum;
    Vector3d P_vect;
    Vector3d e_vect;
    dif_sum.fill(0);
    spe_sum.fill(0);
    P_vect << P->x, P->y, P->z;
    e_vect << LookFrom->x, LookFrom->z, LookFrom->y;

    Vector3d n;
    Vector3d ca, cd, cs;
    n << nv->x, nv->y, nv->z;
    ca << this->ca0, this->ca1, this->ca2;
    cd << this->cd0, this->cd1, this->cd2;
    cs << this->cs0, this->cs1, this->cs2;

    int length = lights.size();
    int i;
    float temp;
    Vector3d lp;
    Vector3d lc;
    Vector3d l_dir;
    Vector3d e_dir;
    Vector3d l_dif;
    Vector3d l_spe;
    Vector3d tempv;
    Vector3d c;

    Point * p = new Point;
    Point * trash = new Point;

    c.fill(1);

    for(i = 0; i < length; i++)
    {
        // Check for intersection with other objects
        shadow->x = lights[i].position[0] - P->x;
        shadow->y = lights[i].position[1] - P->y;
        shadow->z = lights[i].position[2] - P->z;

        bool blocked = false;
        if(i != 0)
        {
            for(int j = 0; j < objects.size(); j++)
            {
                if(this->objnum != j)
                {
                    blocked = blocked || objects[j]->check_shadow(P, shadow, lights[i].position[0], lights[i].position[1], lights[i].position[2]);
                }
            }
        }

        // Transform lights?

        p->x = lights[i].position[0];
        p->y = lights[i].position[1];
        p->z = lights[i].position[2];

        untransform(p, trash,
                    this->xt, this->yt, this->zt,
                    this->rot, this->rx, this->ry, this->rz,
                    this->l, this->w, this->d);

        lp << p->x, p->y, p->z;
        lc << lights[i].color[0], lights[i].color[1], lights[i].color[2];
        lc /= dist;
        lc /= lights[i].attenuation_k;
        l_dir = (lp - P_vect);
        l_dir = l_dir.normalized();
        e_dir = (lc - e_vect);
        e_dir = e_dir.normalized();
        
        temp     = l_dir.dot(n);
        l_dif    = lc * (max(float (0), temp));
        if(!blocked)
            dif_sum += l_dif;
        else
            cerr << "blocked by shadow!\n";
        
        tempv    = e_dir + l_dir;
        tempv    = tempv.normalized();
        temp     = tempv.dot(n);
        l_spe    = lc * pow(max(float (0), temp), this->p);
        if(!blocked)
            spe_sum += l_spe;
    }
    c = c.cwiseMin(ca + dif_sum.cwiseProduct(cd) + spe_sum.cwiseProduct(cs));
    return c;
}

Vector3d Object::ray_trace(Point * origin, Point * direction, Point * r, vector<Object * > objects)
{
    bool verbose = false;
    // Create temp points
    Point * o = new Point;
    Point * d = new Point;

    // Resulting point and normal
    Point * n = new Point;

    // Set point values
    copy(origin, o);
    copy(direction, d);

    // Untransform the temp points
    untransform(o, d, 
                this->xt, this->yt, this->zt,
                this->rot, this->rx, this->ry, this->rz,
                this->l, this->w, this->d);
    
    //o->y = o->z;
    //o->z = o->y;

    normalize(d);

    if(verbose)
    cerr << "\tTracing Ray:\n"
         << "\torigin:      (" << origin->x << ", " << origin->y << ", " << origin->z << ")\n"
         << "\tdirection:   <" << direction->x << ", "<< direction->y << ", " << direction->z << ">\n"
         << "\tt_origin:    (" << o->x << ", " << o->y << ", " << o->z << ")\n"
         << "\tt_direction: <" << d->x << ", " << d->y << ", " << d->z << ")\n";
    // Get intersection
    float intersects = get_intersection(o, d, r, this->e, this->n, 1);

    Point * tr = new Point;
    copy(r, tr);

    // Get normal
    set_normal(tr, n, this->e, this->n);

    //transform(r, n,
    //          this->xt, this->yt, this->zt,
    //          this->rot, this->rx, this->ry, this->rz,
    //          this->l, this->w, this->d);
    normalize(n);

    // Prepare color vector
    Vector3d color;
    // If there was an intersection
    if(intersects != 0)
    {
        // Get the color
        color = this->Lighting(tr, n, objects, o);
        // 1/r^2
    }
    else
    {
        // Otherwise, set color to background color (white)
        color(0) = 0;
        color(1) = 0;
        color(2) = 0;
    }

    // Memory management
    //free(o);
    //free(d);
    //free(r);
    //free(n);

    return color;
}



void get_image(float Fx, float Fy, int Nx, int Ny, vector<Object *> objects)
{

    bool verbose = false;

    // Calculate dFy, dFx
    double long dFx = Fx/Nx;
    double long dFy = Fy/Ny;

    // Calculate basis vectors
    Point * A = new Point;
    Point * B = new Point;
    Point * C = new Point;
    float norm, alpha;

    A->x = LookAt->x - LookFrom->x;
    A->y = LookAt->y - LookFrom->y;
    A->z = LookAt->z - LookFrom->z;
    normalize(A);

    alpha = (A->x * Up->x + A->y * Up->y + A->z * Up->z) / (A->x * A->x + A->y * A->y + A->z * A->z);
    B->x = Up->x - alpha * A->x;
    B->y = Up->y - alpha * A->y;
    B->z = Up->z - alpha * A->z;
    normalize(B);

    C->x = A->y * B->z - A->z * B->y;
    C->y = A->z * B->x - A->x * B->z;
    C->z = A->x * B->y - A->y * B->x;
    normalize(C);

    // Create DFx, DFy vectors
    Point * DFx = new Point;
    Point * DFy = new Point;
    DFx->x = C->x * dFx;
    DFx->y = C->y * dFx;
    DFx->z = C->z * dFx;

    DFy->x = B->x * dFx;
    DFy->y = B->y * dFx;
    DFy->z = B->z * dFx;

    // Calculate start point
    Point * start = new Point;
    start->x = A->x * Fd + (1.0 * Ny / 2) * DFy->x - (1.0 * Nx / 2) * DFx->x;
    start->y = A->y * Fd + (1.0 * Ny / 2) * DFy->y - (1.0 * Nx / 2) * DFx->y;
    start->z = A->z * Fd + (1.0 * Ny / 2) * DFy->z - (1.0 * Nx / 2) * DFx->z;

    Point * vy = new Point;
    Point * vx = new Point;
    Point * p  = new Point;
    Point * r  = new Point;

    Vector3d color, final;
    int length = objects.size();
    float mindist;
    float dist;

    if(verbose);
    cerr << "objects vector size: " << length << "\n";
    // Iterate through film points
    for(int i = 0; i < Ny; i++)
    {
        vy->x = i * DFy->x;
        vy->y = i * DFy->y;
        vy->z = i * DFy->z;
        
        for(int j = 0; j < Nx; j++)
        {
            mindist = 9999999999;
            vx->x = j * DFx->x;
            vx->y = j * DFx->y;
            vx->z = j * DFx->z;
           
            p->x = start->x + vx->x - vy->x;
            p->y = start->y + vx->y - vy->y;
            p->z = start->z + vx->z - vy->z;

            
            int R, G, B;
            for(int k = 0; k < length; k++)
            {
                if(verbose)
                cerr << "Pixel: " << j << "," << i << "\n"
                     << "Point: (" << p->x << ", " << p->y << ", " << p->z << ")\n"
                     << "Object " << k << "\n";
                color = objects[k]->ray_trace(LookFrom, p, r, objects);
                if(color[0] == 0 && color[1] == 0 && color[2] == 0)
                {
                    dist = 9998;
                }
                else
                {
                    dist = distance(LookFrom, r);
                }
                if(r->x == 0 && r->y == 0 && r->z == 0)
                    if(verbose)
                    cerr << "========================================================================================== HERE\n";

                if(verbose)
                cerr << "\t\tColor: " << color[0] << "\t" << color[1] << "\t" << color[2] << "\n" 
                     << "\t\tResult point: " << r->x << ", " << r->y << ", " << r->z << "\n" 
                     << "\t\tdist: " << dist << "\n" 
                     << "\t\tmindist: " << mindist << "\n";
                if (dist < mindist)
                { 
                    if(verbose)
                    cerr << "\t\tNEW MINDIST! Setting...\n";
                    mindist = dist;
                    final[0] = color[0];
                    final[1] = color[1];
                    final[2] = color[2];
                }
            }
            if(verbose)
            cerr << "Putting final color in: " << final[0] << " " << final[1] << " " << final[2] << "\n";
            R = final[0] * 255;
            G = final[1] * 255;
            B = final[2] * 255;
            cout << R << " "
                 << G << " "
                 << B << "\n";
        }
    }
}

int main(int argc, char ** argv)
{

    bool DickNBallz;
    float isqe, isqn, length, width, depth, rotation, rotx, roty, rotz,
          xt, yt, zt, xmin, xmax, ymin, ymax, zmin, zmax;

    if(argc !=  3 && argc !=  6 && argc != 10 &&
       argc != 13 && argc != 19 && argc !=  1)
    {
        cerr << "Incorrect number of inputs: " << argc << "\n";
        cerr << (argc != 3);
        exit(0);
    }

    bool en     = false;
    bool abc    = false;
    bool rot    = false;
    bool trans  = false;
    bool dx     = false;
    bool dy     = false;
    bool dz     = false;
    bool uv     = false;
    if(argc >=  3) {en     = true;};
    if(argc >=  6) {abc    = true;};
    if(argc >= 10) {rot    = true;};
    if(argc >= 13) {trans  = true;};

    if(en)
    {
        isqe = atof(argv[1]);
        isqn = atof(argv[2]);
    }
    else
    {
        isqe = 1;
        isqn = 1;
    }
    if(abc)
    {
        length = atof(argv[3]);
        width  = atof(argv[4]);
        depth  = atof(argv[5]);
    }
    else
    {
        length = 1;
        width  = 1;
        depth  = 1;
    }
    if(rot)
    {
        rotation = atof(argv[6]);
        rotx     = atof(argv[7]);
        roty     = atof(argv[8]);
        rotz     = atof(argv[9]); 
    }
    else
    {
        rotation = 0;
        rotx     = 1;
        roty     = 0;
        rotz     = 0;
    }
    if(trans)
    {
        xt = atof(argv[10]);
        yt = atof(argv[11]);
        zt = atof(argv[12]);
    }
    else
    {
        xt = 0;
        yt = 0;
        zt = 0;
    }

    // Set up defaults
    LookAt->x = 0;
    LookAt->y = 0;
    LookAt->z = 0;
    
    LookFrom->x = 5;
    LookFrom->y = 5;
    LookFrom->z = 5;
    
    Up->x = 0;
    Up->y = 0;
    Up->z = 1;

    vector<Object *> objects;

    if(cin.peek() != '\n')
    {
        int temp;
        cin >> temp;
        if(temp != 2)
        {
            HW6 = temp;
            HW7 = !HW6;
        }
        else
        {
            HW6 = false;
            HW7 = false;
            DickNBallz = true;
        }
    }
    // Load input from cin
    if(cin.peek() != '\n')
    {
        cin >> Nx;
        cin >> Ny;
    }
    if(HW6)
    {
        // Create eye light
        add_light(LookFrom->x, LookFrom->y, LookFrom->z,
                  1,           1,           1,
                  0.1);


        if(cin)
        {
            cin >> LookFrom->x;
            cin >> LookFrom->y;
            cin >> LookFrom->z;
        }

        if(cin)
        {
            cin >> LookAt->x;
            cin >> LookAt->y;
            cin >> LookAt->z;
        }

        if(cin)
        {
            cin >> Up->x;
            cin >> Up->y;
            cin >> Up->z;
        }

        Point_Light l;
        if(cin)
        {
            cin >> l.position[0];
            cin >> l.position[1];
            cin >> l.position[2];
            cin >> l.color[0];
            cin >> l.color[1];
            cin >> l.color[2];
            cin >> l.attenuation_k;

            // Push in user defined light
            lights.push_back(l);
        }
        else
        {
            // Default lights
            add_light(LookFrom->x, LookFrom->y, LookFrom->z + 2,
                      0, 0,     1,
                      0.0008);
        }

        Object * a = new Object(isqe, isqn,
                                length, width, depth,
                                rotation, rotx, roty, rotz,
                                xt, yt, zt,
                                0.2, 0.2, 0.2,
                                0.6, 0.6, 0.6,
                                1.0, 1.0, 1.0,
                                1.0,
                                0);


        objects.push_back(a);
    }
    
    // #7: Assignment 7
    if(HW7)
    {
        LookFrom->x = 12;
        LookFrom->y = 0;
        LookFrom->z = 0;
        //add_light(LookFrom->x, LookFrom->y + 2, LookFrom->z,
        //          0, 0.5,     0,
        //          0.0008);

        //add_light(LookFrom->x, LookFrom->y - 1, LookFrom->z + sqrt(3),
        //          0, 0,     0.5,
        //          0.0008);

        //add_light(LookFrom->x, LookFrom->y - 1, LookFrom->z - sqrt(3),
        //          0.5, 0,     0,
        //          0.0008);

        // Create eye light
        add_light(LookFrom->x, LookFrom->y, LookFrom->z,
                  1,           1,           1,
                  0.02);
        add_light(4, 4, 0,
                  0, 0.7, 0.3,
                  0.01);


        // Parse input
        int num = objects.size();
        Object * a = new Object (2, 2,          // e and n
                                 5, 5, 5,       // scaling
                                 0, 1, 1, 1,    // rotation
                                 -10, 0, 0,      // translation
                                 0.0, 0.3, 0.7, // ambient color
                                 0.3, 0.3, 0.3, // diffuse color
                                 1.0, 1.0, 1.0, // specular color
                                 5.0,           // shininess
                                 num);          // objnum

        num = objects.size();
        Object * b = new Object (2, 2,
                                 .5, .5, .5,
                                 0, 1, 1, 1,
                                 4, 2, 2,
                                 0.5, 0.3, 0.7,
                                 0.3, 0.3, 0.3,
                                 1.0, 1.0, 1.0,
                                 5.0,
                                 num);
                                
        num = objects.size();
        Object * c = new Object (1, 1,
                                 .5, .5, .5,
                                 0, 1, 1, 1,
                                 4, -2, -2,
                                 0.7, 0.3, 0.7,
                                 0.3, 0.3, 0.3,
                                 1.0, 1.0, 1.0,
                                 5.0,
                                 num);
                                
        objects.push_back(a);
        objects.push_back(b);
        objects.push_back(c);

    }
    if(DickNBallz)
    {
        // Dick 'n' Balls
        LookFrom->x = 15;
        LookFrom->y = 0;
        LookFrom->z = 0;
        add_light(LookFrom->x, LookFrom->y, LookFrom->z,
                  1,           1,           1,
                  0.005);

        Object * shaft = new Object(1, 0.1,
                                    0.5, 0.5, 2,
                                    0, 1, 1, 1,
                                    0, 0, 0,
                                    0.9, 0.5, 0.5,
                                    0.3, 0.3, 0.3,
                                    1.0, 1.0, 1.0,
                                    5.0,
                                    0);

        Object * left_ball = new Object(1, 1,
                                        0.75, 0.75, 0.75,
                                        0, 1, 0, 0,
                                        0, -0.7, -2,
                                        0.9, 0.5, 0.5,
                                        0.3, 0.3, 0.3,
                                        1.0, 1.0, 1.0,
                                        5.0,
                                        1);

        Object * right_ball = new Object(1, 1,
                                         0.75, 0.75, 0.75,
                                         0, 1, 0, 0,
                                         0, 0.7, -2,
                                         0.9, 0.5, 0.5,
                                         0.3, 0.3, 0.3,
                                         1.0, 1.0, 1.0,
                                         5.0,
                                         2);

        Object * head = new Object(1, 1.8,
                                   0.6, 0.6, 0.6,
                                   90, 1, 0, 0,
                                   1, 0, 1.8,
                                   0.9, 0.5, 0.5,
                                   0.3, 0.3, 0.3,
                                   1.0, 1.0, 1.0,
                                   5.0,
                                   3);

        objects.push_back(shaft);
        objects.push_back(left_ball);
        objects.push_back(right_ball);
        objects.push_back(head);

    }
    // Compute Fy
    Fy = Ny * Fx / Nx;


    // Get image
    cout << "P3\n" << Nx << " " << Ny << "\n255\n";
    get_image(Fx, Fy, Nx, Ny, objects);

    //cerr << length << "\n";
}

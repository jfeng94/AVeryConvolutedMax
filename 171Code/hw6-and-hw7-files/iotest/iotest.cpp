#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

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

using namespace std;

struct Point_Light
{
    float position[4];
    float color[3];
    float attenuation_k;
};


// Number of points along one dimension of the cube
int N = 20;

// Size of the cube
float xcube_size  = 2.4;
float xmin = -1.2;
float xmax =  1.2;
float ycube_size  = 2.4;
float ymin = -1.2;
float ymax =  1.2;
float zcube_size  = 2.4;
float zmin = -1.2;
float zmax =  1.2;

// Vector to store all N cubed points
vector<Point*> points;

// Shit
int xres, yres;
float length = 1;
float width  = 1;
float depth  = 1;
float rotation = 0;
float rotx = 1;
float roty = 0;
float rotz = 0;
float xt = 0;
float yt = 0;
float zt = 0;
float jitter;
float Nu = 40;
float Nv = 40;
bool tesselate = true;
bool pointdraw = true;
bool wireframe = true;
bool triangles = true;
bool inpoints  = true;
bool outpoints = true;
bool surfacepoints = true;

// Level of jitteriness
float jit_scale = 0.5;

// Superquadric parameters.
float isqe = 1.0;
float isqn = 1.0;

//gl globals
GLUquadricObj *quadratic;
GLuint image;

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

const float x_view_step = 90.0, y_view_step = 90.0;
float x_view_angle = 0, y_view_angle = 0;

bool is_pressed = false;

float cam_position[] = {0, 0, (float) zcube_size + 1.5};

float cam_orientation_axis[] = {1, 1, 1};

float cam_orientation_angle = 0; // degrees

float near_param = 1, far_param = 15,
      left_param = -1, right_param = 1,
      top_param = 1, bottom_param = -1;
vector<Point_Light> lights;

// Function prototypes
void draw_line(float x1, float y1, float z1, float x2, float y2, float z2);
void draw_triangle(float x1, float y1, float z1, 
                   float x2, float y2, float z2,
                   float x3, float y3, float z3);
void updatePoints(void);

// Subroutine for generating random floats from -1.0 to 1.0
float randf()
{
    float n;
    n = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/2)-1;
    return n;
}

void create_lights();
void set_lights();
void init_lights();

// Initialize the OpenGL shit.
void init(void)
{
    create_lights();
    set_lights();
    init_lights();
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(left_param, right_param,
              bottom_param, top_param,
              near_param, far_param);
    glMatrixMode(GL_MODELVIEW); 
}

// Function to handle window resizing.
void reshape(int width, int height)
{
    height = (height == 0) ? 1 : height;
    width = (width == 0) ? 1 : width;
    glViewport(0, 0, width, height);
    mouse_scale_x = (float) (right_param - left_param) / (float) width;
    mouse_scale_y = (float) (top_param - bottom_param) / (float) height;
    glutPostRedisplay();
}

float para_sq_s(float u, float e)
{
    int sign = 1;
    if(sin(u) == 0)
        return 0;
    else
    {
        if(sin(u) < 0)
            sign = -1;
        return sign * pow(abs(sin(u)), e);
    }
}

float para_sq_c(float u, float e)
{
    int sign = 1;
    if(cos(u) == 0)
        return 0;
    else
    {
        if(cos(u) < 0)
            sign = -1;
        return sign * pow(abs(cos(u)), e);
    }
}

void para_sq(float u, float du, float v, float dv)
{
    float x0 = para_sq_c(v * dv, isqn) * para_sq_c(u * du, isqe);
    float y0 = para_sq_c(v * dv, isqn) * para_sq_s(u * du, isqe);
    float z0 = para_sq_s(v * dv, isqn);

    float x1 = para_sq_c((v + 1) * dv, isqn) * para_sq_c(u * du, isqe);
    float y1 = para_sq_c((v + 1) * dv, isqn) * para_sq_s(u * du, isqe);
    float z1 = para_sq_s((v + 1) * dv, isqn);

    float x2 = para_sq_c(v * dv, isqn) * para_sq_c((u + 1) * du, isqe);
    float y2 = para_sq_c(v * dv, isqn) * para_sq_s((u + 1) * du, isqe);
    float z2 = para_sq_s(v * dv, isqn);

    float x3 = para_sq_c((v + 1) * dv, isqn) * para_sq_c((u + 1) * du, isqe);
    float y3 = para_sq_c((v + 1) * dv, isqn) * para_sq_s((u + 1) * du, isqe);
    float z3 = para_sq_s((v + 1) * dv, isqn);

    if(wireframe)
    {
        // Draw lines
        draw_line(x0, y0, z0, x1, y1, z1);
        draw_line(x0, y0, z0, x2, y2, z2);
        draw_line(x1, y1, z1, x2, y2, z2);
        draw_line(x1, y1, z1, x3, y3, z3);
        draw_line(x2, y2, z2, x3, y3, z3);
    }
    if(triangles)
    {
        draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2);
        draw_triangle(x3, y3, z3, x1, y1, z1, x2, y2, z2);
    }
}

void tesselate_obj()
{
    float dv = 2 * M_PI / Nu;
    float du = M_PI / Nv;
    for(int i = 0; i < Nu; i < i++)
    {
        for(int j = 0; j < Nv; i < j++)
        {
            para_sq(i, du, j, dv);
        }
    }
}

void transform()
{
    glTranslatef(xt, yt, zt);
    glRotatef(rotation, rotx, roty, rotz);
    glScalef(length, width, depth);
}
void draw_points()
{
    int n;
    float size;
    float px, py, pz;
    float transp;
    float red, blue;
    Point * p;
    for(int z = 0; z < N; z++)
    {
        for(int y = 0; y < N; y++)
        {
            for(int x = 0; x < N; x++)
            {
                // Get position in vector
                n = z * N * N + y * N + x;
                p = points[n];

                // Get point position
                px = p->x + jit_scale * p->x_jit;
                py = p->y + jit_scale * p->y_jit;
                pz = p->z + jit_scale * p->z_jit;

                // Point inside
                if(p->inside < 0)
                {
                    transp = 0.7;
                    red = 1.0;
                    blue = 0.0;
                    size = 0.008;
                    // Draw the point
                    if(inpoints)
                    {
                        glColor4f(red, 0.0, blue, transp);

                        glPushMatrix();
                        glTranslatef(px, py, pz);
                        glutSolidSphere(size, 10, 10);
                        glPopMatrix();
                    }
                }

                // Point outside
                else if(p->inside  > 0)
                {
                    transp = 0.2;
                    red = 1.0;
                    blue = 0.0;
                    size = 0.008;
                    // Draw the point
                    if(outpoints)
                    {
                        glColor4f(red, 0.0, blue, transp);

                        glPushMatrix();
                        glTranslatef(px, py, pz);
                        glutSolidSphere(size, 10, 10);
                        glPopMatrix();
                    }
                }

                // Point on surface
                else
                {
                    transp = 0.7;
                    red = 0.0;
                    blue = 0.0;
                    size = 0.008;
                    // Draw the point
                    if(surfacepoints)
                    {
                        glColor4f(red, 1.0, blue, transp);

                        glPushMatrix();
                        glTranslatef(px, py, pz);
                        glutSolidSphere(size, 10, 10);
                        glPopMatrix();
                    }
                }
            }
        }
    }
}

void draw_triangle(float x1, float y1, float z1, 
                   float x2, float y2, float z2,
                   float x3, float y3, float z3)
{
    glColor3f(0.33 * (x1/xcube_size) + 0.33,
              0.33 * (y2/ycube_size) + 0.33,
              0.33 * (z3/zcube_size) + 0.33);
    glPushMatrix();
    transform();
    glBegin(GL_TRIANGLES);
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
    glVertex3f(x3, y3, z3);
    glEnd();
    glPopMatrix();
}

void draw_objects()
{
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glRotatef(x_view_angle, 0, 1, 0);
    glRotatef(y_view_angle, 1, 0, 0);

    if(pointdraw)
        draw_points();

    if(tesselate)
        tesselate_obj();
}

void draw_line(float x1, float y1, float z1, float x2, float y2, float z2)
{
    glLineWidth(2);
    glColor4f(0, 0, 0, 1.0);
    glPushMatrix();
    transform();
    glBegin(GL_LINES);
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
    glEnd();
    glPopMatrix();

}

void draw_UTline(float x1, float y1, float z1, float x2, float y2, float z2)
{
    glLineWidth(2);
    glColor4f(0, 0, 0, 1.0);
    glPushMatrix();
    glBegin(GL_LINES);
    glVertex3f(x1, y1, z1);
    glVertex3f(x2, y2, z2);
    glEnd();
    glPopMatrix();

}

GLvoid *font_style = GLUT_BITMAP_9_BY_15;

void HUD()
{
    glColor3f(0,0,0);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(-1, 1.0,-1, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_CULL_FACE);

    glClear(GL_DEPTH_BUFFER_BIT);
    
    string string1 = "t: toggle tesselation    -- OFF";
    string string2 = "    f: toggle faces      -- OFF";
    string string3 = "    w: toggle wireframe  -- OFF";
    string string4 = "i: toggle inside points  -- OFF";
    string string5 = "o: toggle outside points -- OFF";

    if(tesselate)
    {
        string1 = "t: toggle tesselation    -- ON ";
    }
    glRasterPos2f(-0.97,0.95);
    for(string::iterator i = string1.begin(); i != string1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    if(triangles)
    {
        string2 = "    f: toggle faces      -- ON ";
    }
    glRasterPos2f(-0.97,0.90);
    for(string::iterator i = string2.begin(); i != string2.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    if(wireframe)
    {
        string3 = "    w: toggle wireframe  -- ON ";
    }
    glRasterPos2f(-0.97,0.85);
    for(string::iterator i = string3.begin(); i != string3.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    if(inpoints)
    {
        string4 = "i: toggle inside points  -- ON ";
    }
    glRasterPos2f(-0.97,0.80);
    for(string::iterator i = string4.begin(); i != string4.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    if(outpoints)
    {
        string5 = "o: toggle outside points -- ON ";
    }
    glRasterPos2f(-0.97,0.75);
    for(string::iterator i = string5.begin(); i != string5.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    char string6[] = "q: quit";
    glRasterPos2f(-0.97,0.70);
    for(int i = 0; i < (unsigned) strlen(string6); i++)
    {
        glutBitmapCharacter(font_style, string6[i]);
    }

    string eval = "e           : " + to_string(isqe);
    glRasterPos2f(-0.97, -0.85);
    for(string::iterator i = eval.begin(); i != eval.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    string nval = "n           : " + to_string(isqn);
    glRasterPos2f(-0.97, -0.90);
    for(string::iterator i = nval.begin(); i != nval.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    string jval = "jitter scale: " + to_string(jit_scale);
    glRasterPos2f(-0.97, -0.95);
    for(string::iterator i = jval.begin(); i != jval.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    string s1;

    s1 = "z: decrease e by 0.1";
    glRasterPos2f(0.50, 0.95);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "x: decrease e by 0.01";
    glRasterPos2f(0.50, 0.90);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "c: increase e by 0.01";
    glRasterPos2f(0.50, 0.85);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "v: increase e by 0.1";
    glRasterPos2f(0.50, 0.80);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "b: decrease n by 0.1";
    glRasterPos2f(0.50, 0.75);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "n: decrease n by 0.01";
    glRasterPos2f(0.50, 0.70);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "m: increase n by 0.01";
    glRasterPos2f(0.50, 0.65);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = ",: increase n by 0.1";
    glRasterPos2f(0.50, 0.60);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "j: increase jitter by 0.1";
    glRasterPos2f(0.40, -0.90);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    s1 = "k: decrease jitter by 0.1";
    glRasterPos2f(0.40, -0.95);
    for(string::iterator i = s1.begin(); i != s1.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);


}

void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glRotatef(-cam_orientation_angle,
              cam_orientation_axis[0], cam_orientation_axis[1], cam_orientation_axis[2]);
    glTranslatef(-cam_position[0], -cam_position[1], -cam_position[2]);

    // Draw the points/tesselation
    draw_objects();

    // Draw the encompassing box.
    draw_UTline(xmin, ymin, zmin, xmin, ymin, zmax);
    draw_UTline(xmin, ymin, zmin, xmin, ymax, zmin);
    draw_UTline(xmin, ymin, zmin, xmax, ymin, zmin);

    draw_UTline(xmin, ymin, zmax, xmin, ymax, zmax);
    draw_UTline(xmin, ymin, zmax, xmax, ymin, zmax);

    draw_UTline(xmin, ymax, zmin, xmin, ymax, zmax);
    draw_UTline(xmin, ymax, zmin, xmax, ymax, zmin);

    draw_UTline(xmax, ymin, zmin, xmax, ymin, zmax);
    draw_UTline(xmax, ymin, zmin, xmax, ymax, zmin);

    draw_UTline(xmax, ymin, zmax, xmax, ymax, zmax);

    draw_UTline(xmin, ymax, zmax, xmax, ymax, zmax);

    draw_UTline(xmax, ymax, zmin, xmax, ymax, zmax);

    HUD(); 

    glutSwapBuffers();
}

void init_lights()
{
    glEnable(GL_LIGHTING);
    int num_lights = lights.size();
    for(int i = 0; i < num_lights; ++i)
    {
        int light_id = GL_LIGHT0 + i;
        glEnable(light_id);
        glLightfv(light_id, GL_AMBIENT, lights[i].color);
            glLightfv(light_id, GL_DIFFUSE, lights[i].color);
            glLightfv(light_id, GL_SPECULAR, lights[i].color);
        glLightf(light_id, GL_QUADRATIC_ATTENUATION, lights[i].attenuation_k);
    }
}

void set_lights()
{
    int num_lights = lights.size();
    
    for(int i = 0; i < num_lights; ++i)
    {
        int light_id = GL_LIGHT0 + i;
        
        glLightfv(light_id, GL_POSITION, lights[i].position);
    }
}
void mouse_pressed(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        mouse_x = x;
        mouse_y = y;
        is_pressed = true;
    }
    else if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
    {
        is_pressed = false;
    }
}
void mouse_moved(int x, int y)
{
    if(is_pressed)
    {
        x_view_angle += ((float) x - (float) mouse_x) * mouse_scale_x * x_view_step;
        y_view_angle += ((float) y - (float) mouse_y) * mouse_scale_y * y_view_step;

        mouse_x = x;
        mouse_y = y;
        glutPostRedisplay();
    }
}
float deg2rad(float angle)
{
    return angle * M_PI / 180.0;
}
void key_pressed(unsigned char key, int x, int y)
{
    if(key == 'q')
    {
        exit(0);
    }
    else if(key == 't')
        tesselate = !tesselate;
    else if(key == 'w')
        wireframe = !wireframe;
    else if(key == 'f')
        triangles = !triangles;
    else if(key == 'i')
        inpoints = !inpoints;
    else if(key == 'o')
        outpoints = !outpoints;
    else if(key == 'v')
    {
        isqe += 0.1;
        updatePoints();
    }
    else if(key == 'c')
    {
        isqe += 0.01;
        updatePoints();
    }
    else if(key == 'x')
    {
        isqe -= 0.01;
        updatePoints();
    }
    else if(key == 'z')
    {
        isqe -= 0.1;
        updatePoints();
    }
    else if(key == ',')
    {
        isqn += 0.1;
        updatePoints();
    }
    else if(key == 'm')
    {
        isqn += 0.01;
        updatePoints();
    }
    else if(key == 'n')
    {
        isqn -= 0.01;
        updatePoints();
    }
    else if(key == 'b')
    {
        isqn -= 0.1;
        updatePoints();
    }
    else if(key == 'j')
    {
        jit_scale += 0.1;
        updatePoints();
    }
    else if(key == 'k')
    {
        jit_scale -= 0.1;
        updatePoints();
    }

    if(isqe < 0)
        isqe = 0.0001;
    if(isqn < 0)
        isqn = 0.0001;
    if(jit_scale < 0)
        jit_scale = 0;


    glutPostRedisplay();
}
void create_lights()
{
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Light 1 Below
    ///////////////////////////////////////////////////////////////////////////////////////////////
    
    Point_Light light1;
    
    light1.position[0] = -1;
    light1.position[1] = 0;
    light1.position[2] = 1;
    light1.position[3] = 1;
    
    light1.color[0] = 1;
    light1.color[1] = 1;
    light1.color[2] = 1;
    light1.attenuation_k = 0;
    
    lights.push_back(light1);
    
    
}

// Generates all the points and places them into a global vector
void fillPoints(void)
{
    int n;
    Point * p;
    for(int z = 0; z < N; z++)
    {
        for(int y = 0; y < N; y++)
        {
            for(int x = 0; x < N; x++)
            {
                // Get position in vector
                n = z * N * N + y * N + x;
                p = new Point;
                Point * p1 = new Point;
                p->x = (1.0 * x / (N - 1)) * xcube_size + xmin;  
                p->y = (1.0 * y / (N - 1)) * ycube_size + ymin; 
                p->z = (1.0 * z / (N - 1)) * zcube_size + zmin; 
                p->x_jit = randf() * (2.0 / (N - 1));
                p->y_jit = randf() * (2.0 / (N - 1));
                p->z_jit = randf() * (2.0 / (N - 1));

                // Place into vector
                points.push_back(p);
            }
        }
    }
}



// Updates the inside-outsideness of the point when called.
// Should be called every time we change any parameters 
// related to isqn, isqe, and jit_scale.
void updatePoints(void)
{
    int n;
    float px, py, pz;
    Point * p;
    for(int z = 0; z < N; z++)
    {
        for(int y = 0; y < N; y++)
        {
            for(int x = 0; x < N; x++)
            {
                // Trash points
                Point * p1 = new Point;
                Point * p2 = new Point;

                // Get position in vector
                n = z * N * N + y * N + x;
                p = points[n];

                // Get point position
                p1->x = p->x + jit_scale * p->x_jit;
                p1->y = p->y + jit_scale * p->y_jit;
                p1->z = p->z + jit_scale * p->z_jit;

                // Do untransformation
                untransform(p1, p2,
                            xt, yt, zt,
                            rotation, rotx, roty, rotz,
                            length, width, depth);

                // Get inside-outsideness
                p->inside = superquadric(p1->x, p1->y, p1->z, isqe, isqn);
                
            }
        }
    }
}

int main(int argc, char ** argv)
{
    if(argc !=  3 && argc !=  6 && argc != 10 &&
       argc != 13 && argc != 15 && argc != 17 &&
       argc != 19 && argc != 20 && argc != 22 &&
       argc != 1)
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
    bool jitter = false;
    bool uv     = false;
    if(argc >=  3) {en     = true;};
    if(argc >=  6) {abc    = true;};
    if(argc >= 10) {rot    = true;};
    if(argc >= 13) {trans  = true;};
    if(argc >= 15) {dx     = true;};
    if(argc >= 17) {dy     = true;};
    if(argc >= 19) {dz     = true;};
    if(argc >= 20) {jitter = true;};
    if(argc == 22) {uv     = true;};

    if(en)
    {
        isqe = atof(argv[1]);
        isqn = atof(argv[2]);
    }
    if(abc)
    {
        length = atof(argv[3]);
        width  = atof(argv[4]);
        depth  = atof(argv[5]);
    }
    if(rot)
    {
        rotation = atof(argv[6]);
        rotx     = atof(argv[7]);
        roty     = atof(argv[8]);
        rotz     = atof(argv[9]); 
    }
    if(trans)
    {
        xt = atof(argv[10]);
        yt = atof(argv[11]);
        zt = atof(argv[12]);
    }
    if(dx)
    {
        xmin = atof(argv[13]);
        xmax = atof(argv[14]);
        xcube_size = xmax - xmin;
    }
    if(dy)
    {
        ymin = atof(argv[15]);
        ymax = atof(argv[16]);
        ycube_size = ymax - ymin;
    }
    if(dz)
    {
        zmin = atof(argv[17]);
        zmax = atof(argv[18]);
        zcube_size = zmax - zmin;
        cam_position[2] = zcube_size + 1.5;
    }
    if(jitter)
    {
        jit_scale = atof(argv[19]);
    }
    if(uv)
    {
        Nu = atof(argv[20]);
        Nv = atof(argv[21]);
    }
    
    // Create NxNxN vector to keep track of points. 
    fillPoints();

    // Do initial update to figure out inside-outsideness
    updatePoints();

    // Do OpenGL stuff.
    xres = 800;
    yres = 800;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(xres, yres);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("io-test");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse_pressed);
    glutMotionFunc(mouse_moved);
    glutKeyboardFunc(key_pressed);
    glutMainLoop();

    return 0;
}

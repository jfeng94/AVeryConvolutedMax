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

// Point the ray originates from
Point * origin = new Point;
Point * direction = new Point;
Point * result = new Point;
Point * normal = new Point;

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

// Shit
int xres, yres;
float length = 1;
float width = 1;
float depth = 1;
float rotation = 0;
float rotx = 1;
float roty = 1;
float rotz = 1;
float xt = 0;
float yt = 0;
float zt = 0;
float Nu = 40;
float Nv = 40;
bool tesselate = true;
bool pointdraw = true;
bool wireframe = true;
bool triangles = true;
bool demo      = false;
bool autorotate = false;
bool drawend = false;
bool bounding = true;
bool verbose = false;
int ITERATIONS = 100;

// Level of jitteriness
float jit_scale = 0.5;

// Superquadric parameters.
float isqe = 1.0;
float isqn = 1.0;
float eup  = -0.002;
float nup  = 0.007;
float NSCALE = 0.1;

//gl globals
GLUquadricObj *quadratic;
GLuint image;

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

const float x_view_step = 90.0, y_view_step = 90.0;
float x_view_angle = 0, y_view_angle = 0;

bool is_pressed = false;

float cam_position[] = {0, 0, 5};

float cam_orientation_axis[] = {1, 1, 1};

float cam_orientation_angle = 0; // degrees

float near_param = 1, far_param = 10,
      left_param = -1, right_param = 1,
      top_param = 1, bottom_param = -1;
vector<Point_Light> lights;

// Function prototypes
void draw_line(float x1, float y1, float z1, float x2, float y2, float z2);
void draw_triangle(float x1, float y1, float z1, 
                   float x2, float y2, float z2,
                   float x3, float y3, float z3);
void draw_origin();
void draw_end();
void draw_ray();
void set_lights();
void init_lights();
void create_lights();


// Subroutine for generating random floats from -1.0 to 1.0
float randf()
{
    float n;
    n = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/2)-1;
    return n;
}

// Initialize the OpenGL shit.
void init(void)
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_DEPTH_TEST);

    create_lights();
    set_lights();
    init_lights();

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


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                           FUNCTIONS FOR DRAWING                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

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

void para_sq(float u, float du,
             float v, float dv,
             float e, float n)
{
    float x0 = para_sq_c(v * dv, n) * para_sq_c(u * du, e);
    float y0 = para_sq_c(v * dv, n) * para_sq_s(u * du, e);
    float z0 = para_sq_s(v * dv, n);

    float x1 = para_sq_c((v + 1) * dv, n) * para_sq_c(u * du, e);
    float y1 = para_sq_c((v + 1) * dv, n) * para_sq_s(u * du, e);
    float z1 = para_sq_s((v + 1) * dv, n);

    float x2 = para_sq_c(v * dv, n) * para_sq_c((u + 1) * du, e);
    float y2 = para_sq_c(v * dv, n) * para_sq_s((u + 1) * du, e);
    float z2 = para_sq_s(v * dv, n);

    float x3 = para_sq_c((v + 1) * dv, n) * para_sq_c((u + 1) * du, e);
    float y3 = para_sq_c((v + 1) * dv, n) * para_sq_s((u + 1) * du, e);
    float z3 = para_sq_s((v + 1) * dv, n);

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
            para_sq(i, du, j, dv, isqe, isqn);
        }
    }
}



void transform()
{
    glTranslatef(xt, yt, zt);
    glRotatef(rotation, rotx, roty, rotz);
    glScalef(length, width, depth);
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
void draw_objects()
{
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glRotatef(x_view_angle, 0, 1, 0);
    glRotatef(y_view_angle, 1, 0, 0);

    if(tesselate)
        tesselate_obj();

    draw_origin();

    if(drawend)
        draw_end();

    draw_ray();

    //BOUNDING SPHERE
    if(bounding)
    {
        glPushMatrix();
        transform();
        glColor4f(1.0,0,0,0.2);
        glutSolidSphere(sqrt(3), 10, 10);
        glPopMatrix();
    }
}

void draw_origin()
{
    glColor3f(1.0, 0, 0);
    glPushMatrix();
    transform();
    glTranslatef(origin->x, origin->y, origin->z);
    glutSolidSphere(0.1, 10, 10);
    glPopMatrix();
}

void draw_end()
{
    glColor3f(0, 0, 1.0);
    glPushMatrix();
    transform();
    glTranslatef(result->x, result->y, result->z);
    glutSolidSphere(0.1, 10, 10);
    glPopMatrix();
}

void draw_ray()
{
    float x, dx, y, dy, z, dz;
    x = origin->x;
    y = origin->y;
    z = origin->z;

    dx = direction->x;
    dy = direction->y;
    dz = direction->z;

    glLineWidth(2);
    glColor3f(0.0, 0.5, 0.5);
    glPushMatrix();
    transform();
    glBegin(GL_LINES);
    glVertex3f(x, y, z);
    glVertex3f(x + dx, y + dy, z + dz);
    glEnd();
    glPopMatrix();

    if(drawend)
    {
        x = result->x;
        y = result->y;
        z = result->z;

        dx = normal->x;
        dy = normal->y;
        dz = normal->z;

        glLineWidth(2);
        glColor3f(0.0, 0.0, 0.5);
        glPushMatrix();
        transform();
        glBegin(GL_LINES);
        glVertex3f(x, y, z);
        glVertex3f(x + dx, y + dy, z + dz);
        glEnd();
        glPopMatrix();
    }
        
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
    string string4 = "r: toggle auto-rotation  -- OFF";
    string string5 = "d: toggle demo mode      -- OFF";
    string string7 = "s: toggle bounding sphere-- OFF";

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

    if(autorotate)
    {
        string4 = "r: toggle auto-rotation  -- ON ";
    }
    glRasterPos2f(-0.97,0.80);
    for(string::iterator i = string4.begin(); i != string4.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    if(demo)
    {
        string5 = "d: toggle demo mode      -- ON ";
    }
    glRasterPos2f(-0.97,0.75);
    for(string::iterator i = string5.begin(); i != string5.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    if(bounding)
    {
        string7 = "s: toggle bounding sphere-- ON ";
    }
    glRasterPos2f(-0.97,0.70);
    for(string::iterator i = string7.begin(); i != string7.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }



    char string6[] = "q: quit";
    glRasterPos2f(-0.97,0.65);
    for(int i = 0; i < (unsigned) strlen(string6); i++)
    {
        glutBitmapCharacter(font_style, string6[i]);
    }

    string eval = "e           : " + to_string(isqe);
    glRasterPos2f(-0.97, -0.90);
    for(string::iterator i = eval.begin(); i != eval.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    string nval = "n           : " + to_string(isqn);
    glRasterPos2f(-0.97, -0.95);
    for(string::iterator i = nval.begin(); i != nval.end(); i++)
    {
        glutBitmapCharacter(font_style, *i);
    }

    string itval = "Iterations  : " + to_string(ITERATIONS);
    glRasterPos2f(-0.97, -0.85);
    for(string::iterator i = itval.begin(); i != itval.end(); i++)
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

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);


}

void update(void)
{
    int intersect;
    if(autorotate)
        x_view_angle += 1;
    
    if(demo)
    {
        if(isqe > 2)
        {
            eup = -0.002;
        }
        else if(isqe < 0.11)
        {
            eup = 0.002;
        }

        if(isqn > 2)
        {
            nup = -0.007;
        }
        else if(isqn < 0.11)
        {
            nup = 0.007;
        }

        isqe += eup;
        isqn += nup;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
        if(intersect)
        {
            drawend = true;
            set_normal(result, normal, isqe, isqn); 
        }
        else
            drawend = false;
    }

    if(autorotate || demo)
        glutPostRedisplay();
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
    int intersect;
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
    else if(key == 'r')
        autorotate = !autorotate;
    else if(key == 'd')
        demo = ! demo;
    else if(key == 's')
        bounding = !bounding;
    else if(key == '-')
    {
        ITERATIONS--;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == '=')
    {
        ITERATIONS++;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'v')
    {
        isqe += 0.1;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'c')
    {
        isqe += 0.01;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'x')
    {
        if(isqe > 0.1)
            isqe -= 0.01;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'z')
    {
        if(isqe > 0.1)
            isqe -= 0.1;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == ',')
    {
        isqn += 0.1;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'm')
    {
        isqn += 0.01;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'n')
    {
        if(isqn > 0.1)
            isqn -= 0.01;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    else if(key == 'b')
    {
        if(isqn > 0.1)
            isqn -= 0.1;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }

    if(isqe < 0.1)
    {
        isqe = 0.1;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }
    if(isqn < 0.1)
    {
        isqn = 0.1;
        intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    }

    if(intersect)
    {
        drawend = true;
        set_normal(result, normal, isqe, isqn); 
    }
    else
        drawend = false;
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
int main(int argc, char ** argv)
{
    cout << 1 / 1.2  - 1 << "\n";
    if(argc !=  3 && argc !=  6 && argc != 10 &&
       argc != 13 && argc != 15 && argc != 17 &&
       argc != 19 && argc != 21 && argc !=  1)
    {
        cerr << "Incorrect number of inputs: " << argc << "\n"
             << "Usage:\necho (starting point) x y z (direction) x y z |"
             << "./intersection-test "
             << "e n\nlength width depth\ntheta rotx roty rotz\ntx ty tz\n"
             << "Nu Nv\n";
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
    if(argc >= 15) {dx     = true;};
    if(argc >= 17) {dy     = true;};
    if(argc >= 19) {dz     = true;};
    if(argc == 21) {uv     = true;};

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
    if(uv)
    {
        Nu = atof(argv[19]);
        Nv = atof(argv[20]);
    }
    
    // Get ray info from cin
    if(cin)
    {
        cin >> origin->x;
        cin >> origin->y;
        cin >> origin->z;
    }
    else
    {
        origin->x = 5;   
        origin->y = 0.1;
        origin->z = 0.1;
    }

    if(cin)
    {
        cin >> direction->x;
        cin >> direction->y;
        cin >> direction->z;
    }
    else
    {
        direction->x = -1;
        direction->y = 0;
        direction->z = 0;
    }

    // Normalize the direction
    float tx, ty, tz;
    tx = direction->x;
    ty = direction->y;
    tz = direction->z;
    float norm = sqrt(tx * tx + ty * ty + tz * tz);

    direction->x /= norm;
    direction->y /= norm;
    direction->z /= norm;

    // Prepare untransforms on ray
    untransform(origin, direction, xt, yt, zt, rotation, rotx, roty, rotz, length, width, depth);

    cout << "Origin: " << origin->x << "\t" << origin->y << "\t" << origin->z << "\n";
    cout << "Normalized direction vector: " << direction->x << "\t" << direction->y << "\t" << direction->z << "\n";

    if(isqe < 0.1)
    {
        cerr << "For this implementation, only values above 0.1 are allowed for e\n";
        isqe = 0.1;
    }

    if(isqn < 0.1)
    {
        cerr << "For this implementation, only values above 0.1 are allowed for n\n";
        isqn = 0.1;
    }

    int intersect = get_intersection(origin, direction, result, isqe, isqn, NSCALE, ITERATIONS);
    

    if(verbose)
        cout << intersect << "\t" << result->x << "\t" << result->y << "\t" << result->z << "\n";
    if(intersect)
    {
        set_normal(result, normal, isqe, isqn); 
        drawend = !drawend;
    }



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
    glutIdleFunc(update);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse_pressed);
    glutMotionFunc(mouse_moved);
    glutKeyboardFunc(key_pressed);
    glutMainLoop();

    return 0;
}

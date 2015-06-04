#include <vector>
#include <fstream>

#include "superquadric.cuh"
#include "point.cuh"
#include "matrix.cuh"
#include "camera.h"

const char * DELIMITER = " ";

// Set of parser functions to help streamline the ray tracing process
std::vector<Superquadric> parseObjects(const char *filename)
{
    const int MAX_CHARS_PER_LINE = 200;
    const int MAX_TOKENS_PER_LINE = 24;

    std::vector<Superquadric> scene;

    std::ifstream file;
    file.open(filename);

    if (!file.good())
    {
        std::cerr << "Error! File invalid!\n";
        exit(0);
    }

    while(!file.eof())
    {
        char buf[MAX_CHARS_PER_LINE];
        file.getline(buf, MAX_CHARS_PER_LINE);

        int n = 0;
        const char* token[MAX_TOKENS_PER_LINE] = {};
        token[0] = strtok(buf, DELIMITER);

        if (token[0])
        {
            for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
            {
                token[n] = strtok(0, DELIMITER);
                if (!token[n]) break;
            }
        }
        else
            continue;

        //////////////////////////////////
        // Deconstruct line into parts. //
        //////////////////////////////////

        // Translation elements
        float trax, tray, traz;
        trax = atof(token[0]);
        tray = atof(token[1]);
        traz = atof(token[2]);
        Point * tra = new Point(trax, tray, traz);

        // Scaling elements
        float scax, scay, scaz;
        scax = atof(token[3]);
        scay = atof(token[4]);
        scaz = atof(token[5]);
        Point *sca = new Point(scax, scay, scaz);

        // Rotation elements
        float rotx, roty, rotz, theta;
        rotx = atof(token[6]);
        roty = atof(token[7]);
        rotz = atof(token[8]);
        theta = atof(token[9]) * 3.1415926 / 180;
        Point *rot = new Point(rotx, roty, rotz);

        // Eccentricity values
        float E, N;
        E = atof(token[10]);
        N = atof(token[11]);

        // Color diffusion properties
        float difR, difG, difB;
        difR = atoi(token[12]);
        difG = atoi(token[13]);
        difB = atoi(token[14]);
        Point * dif = new Point(difR, difG, difB);

        // Ambient color properties
        float ambR, ambG, ambB;
        ambR = atoi(token[15]);
        ambG = atoi(token[16]);
        ambB = atoi(token[17]);
        Point * amb = new Point(ambR, ambG, ambB);

        // Specular color properties
        float speR, speG, speB;
        speR = atoi(token[18]);
        speG = atoi(token[19]);
        speB = atoi(token[20]);
        Point * spe = new Point(speR, speG, speB);

        // Other light properties
        float  shi,  sne,  opa;
        shi = atof(token[21]);
        sne = atof(token[22]);
        opa = atof(token[23]);

        Superquadric * s = new Superquadric(*tra, *sca, *rot, theta, E, N,
                                            *dif, *amb, *spe, shi, sne, opa);

        scene.push_back(*s);
    }
    return scene;
}

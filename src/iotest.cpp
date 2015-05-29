#include "superquadric.h"
#include "point.h"
#include "matrix.h"

#include <cstdlib>
#include <ctime>
#include <fstream>

int main()
{
    srand(time(0));

    Superquadric * s = new Superquadric();

    std::ofstream out;
    out.open("MatlabTools/iotest_results.txt", std::fstream::out);

    float min_x(-1), max_x(1), min_y(-1), max_y(1), min_z(-1), max_z(1);
    for (int i = 0; i < 10000; i++)
    {
        float x, y, z;
        x = min_x + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_x - min_x)));
        y = min_y + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_y - min_y)));
        z = min_z + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_z - min_z)));

        Point * p = new Point(x, y, z);

        if (s->isq(p) <= 0)
            out << p;
    }

    return 0;
}

#include "superquadric.cpp"
#include "point.cpp"
#include "matrix.cpp"

#include <cstdlib>
#include <ctime>
#include <fstream>

#define EPSILON 1e-2

int main()
{
    srand(time(0));

    Point *rot, *tra, *sca;
    float theta, e, n;


    Superquadric * s  = new Superquadric(1, 1);

    rot              = new Point(1, 0, 0);
    tra              = new Point(0, 0, 0);
    sca              = new Point(0.5, 0.5, 0.5);
    theta             = 0;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(0, 1, 1);
    tra              = new Point(0, 0, 0);
    sca              = new Point(0.5, 0.5, 0.5);
    theta             = 3.1415926 / 2;
    e                 = 0.1;
    n                 = 0.1;
    Superquadric * s3 = new Superquadric(tra, sca, rot, theta, e, n);

    std::ofstream out1;
    out1.open("MatlabTools/iotest_results1.txt", std::fstream::out);
    std::ofstream out2;
    out2.open("MatlabTools/iotest_results2.txt", std::fstream::out);
    std::ofstream out3;
    out3.open("MatlabTools/iotest_results3.txt", std::fstream::out);

    int o1(0), o2(0), o3(0);

    float min_x(-1), max_x(1), min_y(-1), max_y(1), min_z(-1), max_z(1);
    for (int i = 0; i < 3000; i++)
    {
        float x, y, z;
        x = min_x + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_x - min_x)));
        y = min_y + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_y - min_y)));
        z = min_z + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_z - min_z)));

        Point * p = new Point(x, y, z);

        if (0)
        {
            std::cout << "wtf how did you get here?\n";
        }
        else if (std::abs(s->isq(p)) <= EPSILON && o1 < 1000)
        {
            out1 << p;
            o1++;
        }
        else if (std::abs(s2->isq(p)) <= EPSILON && o2 < 1000)
        {
            out2 << p;
            o2++;
        }
        else if (std::abs(s3->isq(p)) <= EPSILON && o3 < 1000)
        {
            out3 << p;
            o3++;
        }
        else
            i--;
    }

    return 0;
}

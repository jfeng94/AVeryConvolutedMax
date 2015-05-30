#include "superquadric.cpp"
#include "point.cpp"
#include "matrix.cpp"

#include <cstdlib>
#include <ctime>
#include <fstream>

int main()
{
    srand(time(0));

    Point *rot, *tra, *sca;
    float theta, e, n;


    Superquadric * s  = new Superquadric(0.01, 0.01);

    rot              = new Point(0, 0, 0);
    tra              = new Point(0, 0, 0);
    sca              = new Point(0.5, 0.5, 0.5);
    theta             = 0;
    e                 = 3;
    n                 = 3;
    Superquadric * s2 = new Superquadric(tra, sca, rot, theta, e, n);

    rot              = new Point(0, 0, 0);
    tra              = new Point(0, 0, 0);
    sca              = new Point(1, 1, 1);
    theta             = 0;
    e                 = 2;
    n                 = 2;
    Superquadric * s3 = new Superquadric(tra, sca, rot, theta, e, n);

    std::ofstream out1;
    out1.open("MatlabTools/iotest_results1.txt", std::fstream::out);
    std::ofstream out2;
    out2.open("MatlabTools/iotest_results2.txt", std::fstream::out);
    std::ofstream out3;
    out3.open("MatlabTools/iotest_results3.txt", std::fstream::out);

    float min_x(-2), max_x(2), min_y(-2), max_y(2), min_z(-2), max_z(2);
    for (int i = 0; i < 5000; i++)
    {
        float x, y, z;
        x = min_x + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_x - min_x)));
        y = min_y + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_y - min_y)));
        z = min_z + static_cast <float> (rand()) /
            (static_cast <float> (RAND_MAX/(max_z - min_z)));

        Point * p = new Point(x, y, z);

        if (s2->isq(p) <= 0)
            out1 << p;
        //else if (std::abs(s2->isq(p)) <= 1e-2)
        //    out2 << p;
        //else if (std::abs(s3->isq(p)) <= 1e-2)
        //    out3 << p;
        else
            i--;
    }

    return 0;
}

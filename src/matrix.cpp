#include "matrix.h"

/*****************************************************************************/
/*                        Rotation Matrix Methods                            */
/*****************************************************************************/
rotMat::rotMat()
{
    this->xyz    = new Point(0, 0, 0);
    this->theta  = 0;
}

rotMat::rotMat(float x, float y, float z, float t)
{
    this->xyz   = new Point(x, y, z);
    this->theta = t;
}

rotMat::rotMat(Point *p, float t)
{
    this->xyz = p;
    this->theta = t;
}

Point * rotMat::apply(Point * p)
{
    // TODO Finish this
    return p;
}

Point * rotMat::unapply(Point * p)
{
    // TODO
    return p;
}

/*****************************************************************************/
/*                         Scaling Matrix Methods                            */
/*****************************************************************************/
scaMat::scaMat()
{
    this->xyz = new Point(1, 1, 1);
}

scaMat::scaMat(float x, float y, float z)
{
    this->xyz = new Point(x, y, z);
}

scaMat::scaMat(Point * p)
{
    this->xyz = p;
}
Point * scaMat::apply(Point * p)
{
    // TODO Finish this
    return p;
}

Point * scaMat::unapply(Point * p)
{
    // TODO
    return p;
}

/*****************************************************************************/
/*                        Translating Matrix Methods                         */
/*****************************************************************************/
traMat::traMat()
{
    this->xyz = new Point(0, 0, 0);
}

traMat::traMat(float x, float y, float z)
{
    this->xyz = new Point(x, y, z);
}

traMat::traMat(Point * p)
{
    this->xyz = p;
}
Point * traMat::apply(Point * p)
{
    // TODO Finish this
    return p;
}

Point * traMat::unapply(Point * p)
{
    // TODO
    return p;
}

// Default constructor
Point::Point()
{
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

// Normal constructor
Point::Point(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

// Returns the norm of a point treated like a vector
Point * Point::norm()
{
    Point * result = new Point();
    *result = *this

    float mag =sqrt(this->x * this->x +
                    this->y * this->y +
                    this->z * this->z);

    result /= mag;

    return result;
}

// OPERATOR OVERLOADS
Point * Point::operator+(Point p)
{
    Point * result = new Point(this->x + p.x,
                               this->y + p.y,
                               this->z + p.z);
    return result;
}

Point * Point::operator-(Point p)
{
    Point * result = new Point(this->x - p.x,
                               this->y - p.y,
                               this->z - p.z);
    return result;
}

Point * Point::operator*(Point p)
{
    Point * result = new Point(this->x * p.x,
                               this->y * p.y,
                               this->z * p.z);
    return result;
}

Point * Point::operator/(Point p)
{
    Point * result = new Point(this->x / p.x,
                               this->y / p.y,
                               this->z / p.z);
    return result;
}

Point * Point::operator+=(Point p)
{
    this->x += p.x;
    this->y += p.y;
    this->z += p.z;

    return this;
}

Point * Point::operator-=(Point p)
{
    this->x -= p.x;
    this->y -= p.y;
    this->z -= p.z;

    return this;
}

Point * Point::operator*=(Point p)
{
    this->x *= p.x;
    this->y *= p.y;
    this->z *= p.z;

    return this;
}

Point * Point::operator/=(Point p)
{
    this->x /= p.x;
    this->y /= p.y;
    this->z /= p.z;

    return this;
}

Point * Point::operator+(float f)
{
    Point * result = new Point(this->x + f,
                               this->y + f,
                               this->z + f)
    return result;
}

Point * Point::operator-(float f)
{
    Point * result = new Point(this->x - f,
                               this->y - f,
                               this->z - f)
    return result;
}

Point * Point::operator*(float f)
{
    Point * result = new Point(this->x * f,
                               this->y * f,
                               this->z * f)
    return result;
}

Point * Point::operator/(float f)
{
    Point * result = new Point(this->x / f,
                               this->y / f,
                               this->z / f)
    return result;
}

Point * Point::operator+=(float f)
{
    this->x += f;
    this->y += f;
    this->z += f;
}

Point * Point::operator-=(float f)
{
    this->x -= f;
    this->y -= f;
    this->z -= f;
}

Point * Point::operator*=(float f)
{
    this->x *= f;
    this->y *= f;
    this->z *= f;
}

Point * Point::operator/=(float f)
{
    this->x /= f;
    this->y /= f;
    this->z /= f;
}

Point * Point::operator=(Point p)
{
    Point result = new Point(p.x, p.y, p.z);
    return result;
}

bool Point::operator==(Point p)
{
    if (this->x == p.x && this->y == p.y && this->z == p.z)
    {
        return true;
    }

    return false;
}

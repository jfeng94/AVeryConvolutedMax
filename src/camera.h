class Camera
{
    private:
        Ray * rayScreen;
        Point * LookFrom, * LookAt;
        Point * e1, *e2, *e3;
        float Fd;
        int Fx, Fy, Nx, Ny;

    public:
        Camera();
        Camera(w, h, Nx, Ny);
};

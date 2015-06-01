# RayTracer
CS179 Project

Changed project to Ray Tracer

Classes we need:
Scene        - Should store the objects, light sources, etc.
Camera       - Stores its physical attributes, location, etc. Member functions
               should essentially be the raytracing kernal call.
Superquadric - Should store its properties. Functions should include a ray
               intersection test.
Rays         - Vector in 3D space, should force to be unit vectors.

Goal 1:
Implement the superquadric class

Quick reminder for myself: The error with the bounding sphere comes from an error in the superquadric class where the z and y axia are flipped. Track where that error is coming from and the image will stop being upside down.

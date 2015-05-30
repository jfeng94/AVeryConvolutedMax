Jerry Feng
CS171
HW 6

*******************************************************************************
Inside - outside test
*******************************************************************************
Nothing special about how to run it. Just call make and put in the parameters:
./io-test e n                               << Eccentricity variables
          length width depth                << Scaling parameters
          theta rx ry rz                    << Rotation parameters
          tx ty tz                          << Translation parameters
          xmin xmax  ymin ymax  zmin zmax   << Point value range
          jitter                            << How "strong" the jittering is
          Nu, Nv                            << Tesselation parameters

For example, ./io-test 1 1  1 2 1 would render an ellipsoid

Extra credit attempt:
- Live testing for e and n values:
    The actual program has functionality for the user to play around with
    e and n.

- Toggle inside points vs outside points:
    The points can be toggled for display both inside and outside
    separately. This can help the user confirm that the points are 
    within the superquadric while speeding up performance
    
- Toggle tesselation object
    The program allows the user to toggle the tesselation object, so as to see
    just the points. This can be helpful in seeing that the inside points
    conform to the superquadric's shape (wireframe only, faces only, all off,
    all on)

- Adjust jitter scale
    the jitter scale can be adjusted mid run to give insight on where the 
    inside-outside boundary lies. Also it looks cool.


*******************************************************************************
Intersection Test
*******************************************************************************
Again, the running method is standard to the specifications in the set:
./intersection-test e n                     << Eccentricity
          length width depth                << Scaling parameters
          theta rx ry rz                    << Rotation parameters
          tx ty tz                          << Translation parameters
          xmin xmax  ymin ymax  zmin zmax   << Point value range
          nu, nv                            << tesselation parameters

Additionally, the program should be called with echo to specify the x, y, z
components of both the starting point and the directional vector respectively.

For instance,
    echo 3 0.1 0.1  -1 0 0 | ./intersection-test 1 1
would have a vector pointing at a sphere, and the intersection point on that
sphere with its normal.

Something to note about the approximation we're using. For e > 1 and x = y = 0,
the gradient function blows up to infinity. The same thing happens for z = 0 at
n > 1. Thus, when using the program, try to avoid having the ray intersect with
the surface at x = 0, y = 0, or z = 0;

Another thing to note:
For our Newton's method, there were instances where the initial iteration would
overshoot the surface and completely miss. To fix this, the first thing I did
was to make sure the directional vectors were always normalized. Secondly, I 
introduced a scaling factor, so instead of
                        t_new = t_old - g(x)/g'(x)
the iterative step now looks like
                        t_new = t_old - scale * g(x)/g'(x)
This may make no mathematical sense, but it has practical uses, since we're
actually approaching the surface more slowly (takes more iterations on average)

Extra credit attempt:
 - Same stuff as in io-test
    Like in io-test, the user can fiddle with e and n and the program will 
    update to reflect any changes in the intersection point. Also tesselation
    toggles.

 - Demo mode
    Values of e and n automatically cycle allowing the user to see the normals
    and intersection point change in real time.

 - Stop at specified points in the iterative process
    The intersection point at different steps in the iterative process can be
    shown by pressing - or + which will change the upper limit of iterations 
    that are allowed to happen.

 - Show the bounding sphere.
    For shiggles.

*******************************************************************************
Ray tracing test (HW6 and HW7)
*******************************************************************************
Same specifications as the set:
./rt-test e n                         << Eccentricity
          length width depth          << Scaling parameters
          theta rx ry rz              << Rotation parameters
          tx ty tz                    << Translation parameters

User specified info should be given using
echo (0 or 1)                         << 1 - hw6. 0 - hw7. 2 - if you’re brave
     Nx Ny                            << Image resolution
     x y z                            << LookFrom paramters
     x y z                            << LookAt parameters
     x y z                            << Up paramters
     x y z                            << Light position
     R G B                            << light color
     k                                << light attenuation constant.

There appears to be some minor issues with lighting. For some reason, it looks
like any translation in the x axis isn't taken into account. Seems like an easy
enough issue to fix, but I can't get it working. I have the suspicion that it's
related to an earlier issue I had where the translations for the bounding 
spheres to get our initial guess would get the y and z axes flipped, so I made
a hackish work around...

Because lighting isn't working properly yet, you won't be able to see my
implementation of shadows working now. I've attached a screen shot of the 
method working when I was still working with bounding spheres. Hopefully, that
plus the code I have in there will be enough for me to warrant even just some 
credit for the shadows part, but if not I understand.

I guess it's important to note that I wrote this part of set 6 to be 
fully compatible with Set 7, so aside from the unimplemented reflections
and refractions, the program is good to go with N objects and M main lights 
with point light shadows. AKA you could totally just grade this part as my
assignment 7 as well.

I didn’t write a parser for assignment 7 because I wanted to get the other
parts working 100 percent before doing that but I ended up being unable to 
do that.

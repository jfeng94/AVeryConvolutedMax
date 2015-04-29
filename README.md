# AVeryConvolutedMaze
CS179 Project

TODO BY WEEK 1:
  Get basics of engine working!
  
    1. MOVEMENT:  Be able to move around using arrow keys. We'll make it a constant rate to start
    2. MAPS:      Be able to import a b/w map and translate black as an impassible point
    3. ENDPOINT:  Recognizes when the point has reached the goal and quit
    4. GAMEOVER:  Your center point collides with a pixel of black. (You should have a radius around the centerpoint that does not allow you to move into black pixels.
Let's use a 640 x 480 map size to make level design easy.

Kernels that we can use:
  1. Gaussian blur. Takes hard edges and essentially makes them traversable
  2. Unsharp mask. Makes blurry regions more traversable
  3. Value Invert. Makes black areas white and white areas black. Greys are also inverted to other shades of grey.
  4. Pixellize. Can be really interesting I think. If you blur then pixellize, all sorts of weird things may happen
  
The antagonist:
I'll call him THE EXTINGUISHER for now. Basically, he wanders around the maze with you, trying to catch you. He emits black rays like a laser range finder which actually serves as a real obstacle. He is also affected by the maze and as you change the maze, he will have to obey the rules. This becomes useful as you can unsharp mask your way through a blurry region and gaussian blur to trap him behind you.

Game mechanics:
  1. If you're not careful, convolute the maze, and find yourself on a black pixel, you lose.
  2. Different levels of grey affect your speed. This is important because you can blur to impede the extinguisher's progress, but your own progress is impeded.
  3. Being impeded causes the extinguisher to enter a fit of rage, and he gains movement speed.

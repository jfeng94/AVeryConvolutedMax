CC = g++ -std=c++11
FLAGS = -g -o

INCLUDE = -I/home/jerry/Dropbox/CS/CS171 -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -I/usr/include/eigen3
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES = ray-trace.cpp *.h
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = rt-test 

all: $(SOURCES)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean


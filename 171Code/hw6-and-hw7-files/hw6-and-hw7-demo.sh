

###################################
echo "Demo of inside-outside test"
###################################


cd iotest
make clean

make

echo "Demonstrate basic executability"
./io-test 1 1

echo "Demonstrate scaling"
./io-test 2 2       1 2 1

echo "Demonstrate rotation"
./io-test 1 0.001   1 1 1   45 0 1 0

echo "Demonstrate translation and redefinition of point range"
./io-test .001 .001 1 1 1   0  1 0 0  1 1 1    0   2      0   2     0   2 

echo "Demonstrate the ability to alter jitter factor"
./io-test 4 4       1 1 1   0  1 0 0  0 0 0   -1.2 1.2   -1.2 1.2  -1.2 1.2  0.2

echo "Demonstrate tesselation variability"
./io-test 4 4       1 1 1   0  1 0 0  0 0 0   -1.2 1.2   -1.2 1.2  -1.2 1.2  0.0 10  10

################################
echo "Demo of intersection-test"
################################

cd ../intersection-test
make clean
make

echo "Basic functionality"
echo 3.0  0.1  0.1   -1.0  0.0  0.0  | ./intersection-test 1 1

echo "Everything at once... to save time..."
echo 3.0 0.1 0.1  -1 0 0 | ./intersection-test 2 2   1 2 1   45 1 0 0   1 1 1

######################
echo "Demo of rt_test"
######################

cd ../ray-trace
make clean
make

echo "Basic functionality"
echo 1 400 400 | ./rt-test | display -

echo "User specified settings"
echo 1 500 500  12 0 0  0 0 0  0 0 1  9 0 5 0 1 0 0.0005 | ./rt-test 1 1  1 1 1  0 1 0 0  0 -2 0 | display -

###########################
echo "Demo of Assignment 7"
###########################

echo "M objects with N light sources, basic"
echo 0 400 400 | ./rt-test | display -

echo "HMMMMMMMMM"
echo 2 400 400 | ./rt-test | display -

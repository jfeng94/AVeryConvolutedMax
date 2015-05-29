figure;
axis equal;
hold on;

file = fopen('iotest_results.txt');
C = fscanf(file, '%f %f %f %f', [3 Inf]);
C = C';

X = C(:, 1);
Y = C(:, 2);
Z = C(:, 3);

for i=1:length(X)
    plot3(X(i), Y(i), Z(i), '.b');
end
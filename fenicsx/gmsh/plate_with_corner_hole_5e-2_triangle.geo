// mesh size

lc = 5e-2;
L = 1;
H = 1;
r = 0.3;

// corner points of the plate

Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

// auxiliary points for the corner hole

Point(31) = {L, H - r, 0, lc};
Point(32) = {L - r, H, 0, lc};

// connect points with lines and arc

Line(1) = {1, 2};
Line(2) = {2, 31};
Line(3) = {32, 4};
Line(4) = {4, 1};

Circle(5) = {31, 3, 32};

// use lines to create a closed curve

Curve Loop(1) = {1, 2, 5, 3, 4};

// asign a surface to the closed curve

Plane Surface(1) = {1};
// Recombine Surface(1) = {1};

// make the surface and curves physical

Physical Curve("bottom") = {1};
Physical Curve("right") = {2};
Physical Curve("top") = {3};
Physical Curve("left") = {4};
Physical Surface("body") = {1};

// mesh

Mesh 2;
Save "plate_with_corner_hole_5e-2_triangle.msh";
// Save "plate_with_corner_hole_5e-2_quad.msh";

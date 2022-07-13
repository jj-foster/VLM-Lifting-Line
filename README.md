LIMIATIONS:

-wings in x-y plane
    -no camber
-no correction for trailing vortices affecting other wings

DATA STRUCTURE:

units: m,N,s

plane
    wing
        -no. chordwise elements (n)
        section
            -leading edge start/end coordinates (x,y)
            -chord start/end (dx)
            -no. spanwise elements (m)

Mesh(n:list,sections:list of lists)
    symmetry is presumed in x-z plane

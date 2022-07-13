### Limitations:

- Wings in x-y plane so no camber.
- Currently only one wing supported.

### Data Structure:
```
units: m,N,s

plane
    wing
        -no. chordwise elements (n)
        section
            -leading edge start/end coordinates (x,y)
            -chord start/end (dx)
            -no. spanwise elements (m)

Mesh(n:list,sections:list[list])
    symmetry is presumed in x-z plane
```

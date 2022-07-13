import json
import numpy as np
from matplotlib import pyplot as plt
from numba import jit, config

from mesh_generator import Mesh

class PlaneDefinitionError(Exception):
    """Raised when plane input data is formatted incorrectly."""
    def __init__(self,error):
        print(error)
        print(
"""PlaneDefinitionError: Plane input data formatted incorrectly. See standard forat below: 
{
    "plane":{
        "wing":[    #   Multiple wings can be defined.
            {
                "n":int,    #   No. chordwise panels.
                "section":[ #   Multiple wing sections can be defined.
                    {
                        "m":int,    #   No. spanwise panels
                        "leading_edges":[start,end],
                        "chords":[start,end]
                    }
                ]
            }
        ]
    }
}"""
    )
        exit()

class VLM():
    def __init__(self):
        self.points=[]
        self.Panels=[]

        self.rho=1.225

    def geometry(self,file:str=True,data:dict=False):
        """
        Reads geometry file or accepts json data.
        Checks geometry.
        Generates mesh.

        Parameters
        ----------
        file : str
               Json file with plane data.
        data : dict
               Json style dict with plane data.
        """
        ## Get input data
        if data==False:
            if file.split('.')[1]!="json":
                raise FileNotFoundError(f"{file} not in .json format")

            with open(file,'r') as f:
                plane=json.load(f)
        else:
            plane=data

        ## Assign variables
        # CURRENTLY ONLY 1 WING SUPPORTED!!
        try:
            self.S_ref=plane["plane"]["S_ref"]
            self.b_ref=plane["plane"]["b_ref"]
            self.c_ref=plane["plane"]["c_ref"]

            n=plane["plane"]["wing"][0]["n"]
            m=[x["m"] for x in plane["plane"]["wing"][0]["section"]]
            leading_edges=[x["leading_edges"] for x in plane["plane"]["wing"][0]["section"]]
            chords=[x["chords"] for x in plane["plane"]["wing"][0]["section"]]
        
        ## Checking inputs
        except KeyError as error:
            raise PlaneDefinitionError(error)
        except json.decoder.JSONDecodeError as error:
            raise PlaneDefinitionError(error)

        if plane["plane"]["wing"]==[]:
            raise PlaneDefinitionError("No wing defined.")

        for wing in plane["plane"]["wing"]:
            if wing["section"]==[]:
                raise PlaneDefinitionError("No wing section defined.")
            for section in wing["section"]:
                if len(section["leading_edges"])!=len(section["chords"]):
                    raise PlaneDefinitionError("Number of leading edges do not match chords.")

        ## Generate mesh
        mesh=Mesh(n=n,m=m,leading_edges=leading_edges,chords=chords)
        self.points=mesh.calc_points()
        self.Panels=mesh.calc_panels()

        return None

    def plot_mesh(self,cp=False):
        fig,ax=plt.subplots()

        cp_x=[]
        cp_y=[]
        for panel in self.Panels:
            grid_x=[panel.P1[0],panel.P2[0],panel.P3[0],panel.P4[0]]
            grid_y=[panel.P1[1],panel.P2[1],panel.P3[1],panel.P4[1]]
            
            if max(grid_y)>0:
                ax.plot(grid_x,grid_y,color='k')
            else:
                ax.plot(grid_x,grid_y,color='gray')

            cp_x.extend([panel.cp[0]])
            cp_y.extend([panel.cp[1]])

        if cp==True:
            ax.scatter(cp_x,cp_y,color='r',s=10)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')

        ax.set_ylim(0)

        #plt.show()
        
        return plt

    def vlm(self,Q_inf_mod:float,alpha:float,beta:float):
        """
        Computes vortex strength at each collocation point on the mesh.\n
        Solves the following equation:\n
        [a_ij][gamma_i]=-V_inf.n_i where a_ij=(u,v,w)_ij.n_i

        See 'Low Speed Aerodynamics - From Wing Theory to Panel Methods - Joseph Katz & Allen Plotkin' for more detail.

        Parameters
        ----------
        |Q_inf|, alpha, beta : float
            |Q_inf| - absolute free stream velocity (m/s)
            alpha - angle of attack (deg)
            beta - sideslip angle of attack (deg)
        """

        @jit(nopython=True)
        def line_vortex(x,y,z,x1,y1,z1,x2,y2,z2,gamma,R):
                """
                Calculates induced velocity on a point due to a line vortex element.
                """
                
                r1x2_x=(y-y1)*(z-z2)-(z-z1)*(y-y2)
                r1x2_y=-(x-x1)*(z-z2)-(z-z1)*(x-x2)
                r1x2_z=(x-x1)*(y-y2)-(y-y1)*(x-x2)

                r1x2_mod2=r1x2_x**2+r1x2_y**2+r1x2_z**2

                r1_mod=np.sqrt((x-x1)**2+(y-y1)**2+(z-z1)**2)
                r2_mod=np.sqrt((x-x2)**2+(y-y2)**2+(z-z2)**2)

                # Singularity condition:
                # If point lies on vortex, induced velocities=0
                if r1_mod<R or r2_mod<R or r1x2_mod2<R:
                    u=v=w=0.0
                    return np.array([u,v,w])

                r0dotr1=(x2-x1)*(x-x1)+(y2-y1)*(y-y1)+(z2-z1)*(z-z1)
                r0dotr2=(x2-x1)*(x-x2)+(y2-y1)*(y-y2)+(z2-z1)*(z-z2)

                K=(gamma/(4*np.pi*r1x2_mod2))*((r0dotr1/r1_mod)-(r0dotr2/r2_mod))
                
                u=K*r1x2_x
                v=K*r1x2_y
                w=K*r1x2_z

                return np.array([u,v,w])

        @jit(nopython=True)
        def horseshoe_vortex(x,y,z,x_A,y_A,z_A,x_B,y_B,z_B,
                            x_C,y_C,z_C,x_D,y_D,z_D,gamma,R):
            """
            Calculates induced velocity on a point due to a horseshoe vortex.
            """
            
            
            #   Induced velocities by each line vortex in the trailing vortex
            q1=line_vortex(x,y,z,x_A,y_A,z_A,x_B,y_B,z_B,gamma,R)
            q2=line_vortex(x,y,z,x_B,y_B,z_B,x_C,y_C,z_C,gamma,R)
            q3=line_vortex(x,y,z,x_C,y_C,z_C,x_D,y_D,z_D,gamma,R)

            q=q1+q2+q3  #   Induced velocity by horseshoe vortex on point P
            _q=q1+q3    #   Induced downwash velocity by horseshoe vortex on point P

            return q,_q

        @jit(nopython=True)
        def loop(a,b,RHS,N,alpha,panels_n,panels_cp,panels_B,panels_C,Q_inf):   
            for i in range(N):  #   collocation point loop
                n=panels_n[i]#.calc_normal(alpha)  #   panel normal vector
                RHS[i]=np.dot(-Q_inf,n)         #   velocity vector normal to panel

                for j in range(N):  #   vortex element loop   
                    x,y,z=panels_cp[i]
                    x_B,y_B,z_B=panels_B[j]
                    x_C,y_C,z_C=panels_C[j]

                    x_A=x_D=20*b_ref
                    y_A=y_B
                    y_D=y_C
                    z_A=x_A*np.sin(alpha)
                    z_D=x_D*np.sin(alpha)

                    q,_q=horseshoe_vortex(
                        x,y,z,x_A,y_A,z_A,x_B,y_B,z_B,
                        x_C,y_C,z_C,x_D,y_D,z_D,1,R
                    )
                    # Wing symmetrical in x-z plane.
                    q_image,_q_image=horseshoe_vortex(
                        x,-y,z,x_A,y_A,z_A,x_B,y_B,z_B,
                        x_C,y_C,z_C,x_D,y_D,z_D,1,R
                    )

                    q=np.array([q[0]+q_image[0],q[1]-q_image[1],q[2]+q_image[2]])
                    _q=np.array([_q[0]+_q_image[0],_q[1]-_q_image[1],_q[2]+_q_image[2]])
                    
                    a[i][j]=np.dot(q,n)     #   influence coefficient matrix
                    b[i][j]=np.dot(_q,n)    #   normal component of wake induced downwash

            return a,b

        panels=self.Panels
        rho=self.rho
        b_ref=self.b_ref
        alpha=np.deg2rad(alpha)
        beta=np.deg2rad(beta)

        Q_inf=Q_inf_mod*np.array([
            np.cos(alpha)*np.cos(beta),
            -np.sin(beta),
            np.sin(alpha)*np.cos(beta)
        ])

        R=1e-10

        N=len(panels)
        a=np.zeros([N,N])
        RHS=np.zeros([N])
        b=np.zeros([N,N])
        
        panels_n=np.array([panel.n for panel in panels])
        panels_cp=np.array([panel.cp for panel in panels])
        panels_B=np.array([panel.B for panel in panels])
        panels_C=np.array([panel.C for panel in panels])

        a,b=loop(a,b,RHS,N,alpha,panels_n,panels_cp,panels_B,panels_C,Q_inf)  #   main vlm loop

        gamma=np.linalg.solve(a,RHS)
        w_ind=np.matmul(b,gamma)

        ## Aero force computation
        for j in range(N):
            panels[j].gamma=gamma[j]
            panels[j].w_jnd=w_ind[j]
            
            panels[j].dL=rho*Q_inf_mod*gamma[j]*panels[j].dy
            panels[j].dD=-rho*w_ind[j]*gamma[j]*panels[j].dy
        
        L=round(2*sum([panel.dL for panel in panels]),5)    #   x2 for symmetry
        Di=round(2*sum([panel.dD for panel in panels]),5)

        ## Aero coefficients

        C_L=round(L/(0.5*rho*self.S_ref*Q_inf_mod**2),5)
        C_Di=round(Di/(0.5*1.225*self.S_ref*Q_inf_mod**2),5)

        return C_L, C_Di
        #return L, Di

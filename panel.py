import numpy as np

class Panel():
    def __init__(self,P1,P2,P3,P4):
        self.P1=P1
        self.P2=P2
        self.P3=P3
        self.P4=P4

        self.S=self.calc_area()
        self.cp=self.calc_cp()

        self.B,self.C=self.calc_bound_vortex()

        self.dy=np.linalg.norm(self.P1-self.P2)
        self.n=None
        self.gamma=None
        self.dL=0
        self.w_ind=None
        self.dD=0

    def calc_area(self):
        """
        Calculates area of panel. Assumes quadrelateral is convex and consists of
        2 equally sized triangles.
        """
        P12=self.P2-self.P1
        P23=self.P3-self.P2
        P34=self.P4-self.P3
        P41=self.P1-self.P4

        S=0.5*np.linalg.norm(np.cross(P12,P23)+np.cross(P34,P41))

        return S

    def calc_cp(self):
        """
        Calculates panel collocation point. This is the point on the panel where
        the velocity induced by vortex elements acts. It lies at 3/4 chord - see
        Katz & Plotkin for details.
        """
        c0=np.linalg.norm(self.P4-self.P1)
        c1=np.linalg.norm(self.P3-self.P2)
        
        _P1=self.P1+(c0*0.75,0,0)
        _P2=self.P2+(c1*0.75,0,0)

        _P12=_P2-_P1

        cp=0.5*_P12+_P1

        return cp

    def calc_bound_vortex(self):
        c0=np.linalg.norm(self.P4-self.P1)
        c1=np.linalg.norm(self.P3-self.P2)
        
        B=self.P1+(c0*0.25,0,0)
        C=self.P2+(c1*0.25,0,0)

        return B,C

    def calc_normal(self,alpha):
        P1cp=self.cp-self.P1
        P2cp=self.cp-self.P2
        
        n=np.cross(P2cp,P1cp)
        n=n/np.linalg.norm(n)
        
        self.n=n

        return self.n

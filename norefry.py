""" Module for NoReFry algorithm """

# python3 -m pip install matplotlib
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_colorbar_with_padding(ax):
    """
    Create colorbar axis that fits the size of a plot
    detailed here: http://chris35wills.github.io/matplotlib_axis/
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return(cax) 

def circle(r,x0=0,y0=0,n=50):

    t = np.linspace(0,2*np.pi,n)
    x = x0 + r*np.cos(t)
    y = y0 + r*np.sin(t)
    return x,y

def getPlaneABC(P1,P2,P3):
    """
    Calculate plane parameters from three points
    Plane is defined as: z(x,y) = A*x + B*y + C
    points = [[   P1   ],[   P2   ],[   P3   ]]
    points = [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]

    Example:
        `getPlaneABC(np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]))`
    """
    
    u = P2 - P1         # First vector in the plane
    v = P3 - P1         # Second vector in the plane
    
    n = np.cross(u,v)   # Orthogonal vector to the plane

    if n[2]==0: raise Exception(f"Points cannot be in one line! u={u}")

    #  = n.(u-v) = (n1,n2,n3).((u1,u2,u3)-(v1,v2,v3))  ...and so on

    A = -n[0]/n[2]
    B = -n[1]/n[2]
    C = (n[0]*P1[0]+n[1]*P1[1]+n[2]*P1[2]) / n[2]

    return A, B, C


def planeLevel(data,points,radius,plot=False):
    """ Fit plane and level data """

    # TODO: Test this function
    sX,sY = data.shape

    print(data.shape)

    # Append empty column to points (used for z coordinate)
    points = np.append(points,np.zeros((3,1)),axis=1)

    # Calculate z coordinate for each point (mean value within given radius)
    for p in range(3):
        z = 0   # sum of data
        n = 0   # number of involved points
        pX = points[p][0].astype(int)
        pY = points[p][1].astype(int)
        for x in range(pX-radius,pX+radius):
            for y in range(pY-radius,pY+radius):
                if (((pX-x)**2+(pY-y)**2) < radius and
                    x>=0 and y>=0 and x<sX and y<sY):
                    z += data[x][y]
                    n += 1
        if n==0: raise Exception(f"Point {p} ({pX},{pY}) out of image!")
        points[p][2] = z/n      # Calculate mean z

    # Calculate plane from points
    A,B,C = getPlaneABC(points[0],points[1],points[2])
    X,Y = np.meshgrid(np.linspace(0,sY-1,sY),np.linspace(0,sX-1,sX))
    plane = A*X + B*Y + C

    # Subtract plane from the data
    levelledData = data - plane

    # Plot output
    if plot:

        fig=plt.figure(figsize=(10,3))
        fig.canvas.set_window_title('Level data using three points')


        ax1=fig.add_subplot(131)
        plt.imshow(data.T,origin='lower',vmin=np.min(data), vmax=np.max(data))
        plt.title('Original data')
        for point in points:
            x,y = circle(radius,x0=point[0],y0=point[1])
            plt.plot(x,y,color='k',linestyle='--')
        plt.xlim(0,sX)
        plt.ylim(0,sY)
        cax1=make_colorbar_with_padding(ax1)
        b1 = plt.colorbar(cax=cax1)
        # fig.subplots_adjust(right=2)

        ax2=fig.add_subplot(132)
        plt.imshow(plane.T,origin='lower')
        plt.title('Fitted plane')
        cax2=make_colorbar_with_padding(ax2)
        b2 = plt.colorbar(cax=cax2)

        ax3=fig.add_subplot(133)
        plt.imshow(levelledData.T,origin='lower')
        plt.title('Levelled data')
        cax3=make_colorbar_with_padding(ax3)
        b3 = plt.colorbar(cax=cax3)
        
        plt.tight_layout()
        plt.show()

    return levelledData

filename = 'data/#75_v6_h8.dat'

data = (-np.loadtxt(filename)*1e9).T
data = planeLevel(data,np.array([[80,80],[560,80],[80,400]]),80,plot=False)
data[data<0] = 0

# plt.imshow(data.T); plt.show()

sX,sY = data.shape
d0 = -np.sort(-np.reshape(data,sX*sY))

plt.plot(d0)
plt.show()


""" Module for NoReFry algorithm 

**Notes to data dimensions and plotting:**
    ``Numpy`` loads text file of `M` rows and `N` columns to `MxN` matrix.
    The first coordinate would therefore indicate rows, i.e. `y` direction.
    In order to access data as `data[x][y]`, data are transposed upon reading.
    ``Matplotlib`` plotting command `pyplot.imshow` however prints the first
    coordinate (`x`) in vertical direction. Therefore, second transposition is
    used to plot output.

    data = transpose(file with matrix of M rows and N columns)
    size of data = M x N
    data[x,y]; where x = [0,M), y = [0,N)
    plt.imshow(data.T) so x-axis is [0,M) and y-axis is [0,N)
"""

# %%
# %matplotlib widget

import os
import numpy as np
from numpy.lib.function_base import interp
from scipy import interpolate
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define functions -------------------------------------------------------------

def make_colorbar_with_padding(ax):
    """
    Create colorbar axis that fits the size of a plot detailed here:
    http://chris35wills.github.io/matplotlib_axis/
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    return(cax) 

def circle(r,x0=0,y0=0,n=50):
    """ Get (x,y) coordinates of a circle. """

    t = np.linspace(0,2*np.pi,n)
    x = x0 + r*np.cos(t)
    y = y0 + r*np.sin(t)
    return x,y

def getPlaneABC(P1,P2,P3):
    """
    Calculate plane parameters from three points.
    Plane is defined as: z(x,y) = A*x + B*y + C.
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
    """ Level data by subtracting plane defined by three points. """

    sX,sY = data.shape  # Get dimensions of data

    # Append empty column to points (used for `z` coordinate)
    points = np.append(points,np.zeros((3,1)),axis=1)

    # Calculate `z` coordinate for each point (mean value within given radius)
    for p in range(3):
        z = 0   # sum of data
        n = 0   # number of involved points
        pX = points[p,0].astype(int)
        pY = points[p,1].astype(int)
        for x in range(pX-radius,pX+radius):
            for y in range(pY-radius,pY+radius):
                if (((pX-x)**2+(pY-y)**2) < radius and
                    x>=0 and y>=0 and x<sX and y<sY):
                    z += data[x,y]
                    n += 1

        # Check if current point was within the data
        if n==0: raise Exception(f"Point {p} ({pX},{pY}) out of image!")
        points[p,2] = z/n      # Calculate mean `z` and store it into `points`

    # Calculate plane from points
    A,B,C = getPlaneABC(points[0],points[1],points[2])
    X,Y = np.meshgrid(np.linspace(0,sX-1,sX),np.linspace(0,sY-1,sY))
    plane = (A*X + B*Y + C).T

    # Subtract plane from the data
    levelledData = data - plane

    # Plot output
    if plot:

        # Init new figure
        fig=plt.figure(figsize=(10,3))
        fig.canvas.set_window_title('Level data using three points')

        # Plot original data (input)
        ax1=fig.add_subplot(131)
        plt.imshow(data.T,origin='lower',vmin=np.min(data), vmax=np.max(data))
        plt.title('Original data (input)')
        for point in points:
            x,y = circle(radius,x0=point[0],y0=point[1])
            plt.plot(x,y,color='k',linestyle='--')
        plt.xlim(0,sX)
        plt.ylim(0,sY)
        cax1=make_colorbar_with_padding(ax1)
        b1 = plt.colorbar(cax=cax1)

        # Plot fitted plane
        ax2=fig.add_subplot(132)
        plt.imshow(plane.T,origin='lower')
        plt.title('Fitted plane')
        cax2=make_colorbar_with_padding(ax2)
        b2 = plt.colorbar(cax=cax2)

        # Plot levelled data (output)
        ax3=fig.add_subplot(133)
        plt.imshow(levelledData.T,origin='lower',vmin=0,vmax=np.max(levelledData))
        plt.title('Levelled data (output)')
        cax3=make_colorbar_with_padding(ax3)
        b3 = plt.colorbar(cax=cax3)
        
        plt.tight_layout()
        plt.show()

    return levelledData


# Load files & create depth scan

filenames = ['data/test_data_1.dat',
             'data/test_data_2.dat',
             'data/test_data_3.dat',
             'data/test_data_4.dat',
             'data/test_data_5.dat',
             'data/test_data_6.dat',
             'data/test_data_7.dat']

energies = np.array([4227.57,6124.28,10182.96,16717.71,25699.32,45753.05,74263.18])/1000

d0 = None

for filename in filenames:

    data = (-np.loadtxt(filename)*1e9).T
    data = planeLevel(data,np.array([[50,50],[50,400],[600,250]]),100,plot=True)
    # print(np.min(data),np.max(data))
    data[data<5] = 0

    sX,sY = data.shape
    depth_scan = -np.sort(-np.reshape(data,sX*sY))  # Sort data descending
    N = sX*sY

    # N = 1000
    # f = interpolate.interp1d(np.arange(0,sX*sY),depth_scan)
    # x2 = np.arange(0,N)*sX*sY/N
    # depth_scan = f(x2)

    if d0 is None:
        d0 = depth_scan.reshape(1,N)
    else:
        d0 = np.append(d0,depth_scan.reshape(1,N),axis=0)

    # break

dx = dy = 0.0022

# Cropping of negative data causes there are many redundant zeros in `d0` ->
# -> get index of last non-zero element and crop `d0` to new size
d0N = np.max(np.nonzero(d0))+1
d0 = d0[:,0:d0N]

d0[-1,0] = d0[-1,1]
d0[-2,0:1] = d0[-2,2]

# X-vector (area) of D-scan
S0 = np.arange(0,d0N)*dx*dy

# Plot depth scan
fig = plt.figure()
for i in range(len(d0)):
    plt.plot(S0,d0[i],label=str(i))
plt.title('Depth scan')
plt.xlabel(r'S [mm$^2$]')
plt.ylabel('Depth [nm]')
plt.legend()
plt.show()


# %%
# Initialize matrices ----------------------------------------------------------

# Size in depth-direction, step, axis vector
nD = 200; dD = np.max(d0)/(nD-1); d_axis = np.arange(0,nD)*dD
# Size in area-direction, step, axis vector
nS = 200; dS = np.max(S0)/(nS-1); s_axis = np.arange(0,nS)*dS

# Depth scan matrix (`DS`): It is identical to the depth scan but shaped as
# matrix. Values of the pixels are pulse energies corresponding to the given
# imprint.
DS = np.zeros((nD,nS))
# Inverse depth scan matrix (`iDS`): Same as `DS` but values are 1/energy.
iDS = np.zeros((nD,nS))

print("Matrix initialization: ",end='')
# Fill matrices with numbers
for i in range(len(d0)):
    print(i,end='')
    for d in range(d0N):
        id = np.round(d0[i,d]/dD).astype(int)   # Index of depth
        iS = np.round(S0[d]/dS).astype(int)     # Index of area
        if id>=0 and iS>=0:
            DS[id,iS]  = energies[i]
            iDS[id,iS] = 1/energies[i]
print('')


# Remove gaps in columns (important for large matrices)
for iS in range(nS):
    for id in range(nD):
        if DS[id,iS]>0:
            e = DS[id,iS]
            for id2 in range(id,nD):
                if DS[id2,iS] == e:
                    for id3 in range(id,id2):
                        DS[id3,iS]  = e
                        iDS[id2,iS] = 1/e

fig = plt.figure()
plt.imshow(DS,origin='lower',aspect='auto',interpolation='none',
    extent =[s_axis.min(), s_axis.max(), d_axis.min(), d_axis.max()])
plt.title('DS matrix')
plt.xlabel(r'S [mm$^2$]')
plt.ylabel('Depth [nm]')
plt.show()


# %%
# Init f-scan and dose vector --------------------------------------------------

DS_count = (DS>0)

E0 = (DS[:,0]).copy()       # just to keep original value of E
E = E0.copy()               # E is a dose vector
E2 = E**2                   # variance of E
E_count = (E>0)           # nonzero E-counter
E_avg = E.copy()            # averaged dose vector
E_avg2 = E2.copy()          # variance of averaged E vector
E_avg_count = 1*(E_avg>0)   # counter of non-negative values of E_avg
E_sigma = np.sqrt(E2-E**2)  # sigma vector 

f = np.zeros(nS)            # fluence scan vector
f2 = f**2                   # variance of f
f_count = f.copy()          # counter of f-scan
f_avg = f.copy()            # averaged f vector
f_avg2 = f2.copy()          # variance of averaged f vector
f_avg_count = (f_avg>0)   # counter of non-negative values of f_avg
f_sigma = np.sqrt(f2-f**2)  # sigma vector

print("DS",DS.shape," | E",E.shape, " | f",f.shape, " | f_c",f_count.shape)

fig = plt.figure()
plt.plot(d_axis,E0)

# * Iteraction loop ------------------------------------------------------------

for i in range(10):

    df = E_avg.dot(iDS)
    f = f + df
    f_count = f_count + E_avg_count.dot(DS_count)
    f_avg[f_count>0] = f[f_count>0] / f_count[f_count>0]

    f_avg_count = 1*(f_avg>0)

    # f2 = f2 + (E_avg**2).dot(iDS**2)
    # f_avg2[f_count>0] = f2[f_count>0] / f_count[f_count>0]
    # f_sigma = np.sqrt(f_avg2 - f_avg**2)

    # Normalize `f_avg`
    f_avg = (f_avg-np.min(f_avg))/(np.max(f_avg)-np.min(f_avg))
    f_avg[0] = 1

 
    dE = DS.dot(f_avg)
    E = E + dE
    E_count = E_count + DS_count.dot(f_avg_count)
    E_avg[E_count>0] = E[E_count>0] / E_count[E_count>0]

    E_avg_count = 1*(E_avg>0)

    # E2 = E2 + (DS**2).dot(f_avg**2)
    # E_avg2[E_count>0] = E2[E_count>0] / E_count[E_count>0]
    # E_sigma = np.sqrt(E_avg2 - E_avg**2)


    plt.plot(d_axis,E_avg)

# Do first plot here -------
# fig = plt.figure()
# plt.scatter(d_axis,E_avg-E_sigma/2)
# plt.scatter(d_axis,E_avg+E_sigma/2)
# plt.scatter(d_axis,E_avg)
plt.show()

print("DS",DS.shape," | E",E.shape, " | f",f.shape)


# %%

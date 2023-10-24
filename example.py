

# %% ================================ Import ===================================

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import norefry_library as nrfl

# %% ================== Initialization of input data ===========================

# List of filenames with full or relative path
filenames = ['data/test_data_1.dat',
             'data/test_data_2.dat',
             'data/test_data_3.dat',
             'data/test_data_4.dat',
             'data/test_data_5.dat',
             'data/test_data_6.dat',
             'data/test_data_7.dat']

# Energies corresponding to provided data
energies = np.array([4.227,6.124,10.182,16.717,25.699,45.753,74.263])

width = 1.392    # Real width of data [mm]
height = 1.044   # Real height of data [mm]

# %% ================= Load data and create d-scan matrix ======================

# Init d-scan matrix which columns will be individual depth scans
# Dimensions are not know yet -> init as `None`.
d0 = None

# Optional: Downsample depth scans. Definition of `DS` matrix can be realy
# time-consuming in case of large input images. Set to `None` if not required.
# It defaults to image width x height, i.e. 640*480 = 307200 in this example.
N = None

# Fill d-scan matrix with data loaded from extern files
for filename in filenames:

    # Load data from file
    data = (-np.loadtxt(filename)*1e9).T

    # Remove linear background by providing three points and radius
    points = np.array([[50,50],[50,400],[600,250]])
    data = nrfl.removeLinearBackground(data, points, radius=100, plot=False)
    data[data<5] = 0    # Remove data below certain threshold

    nX,nY = data.shape  # Get dimensions of data
    depth_scan = -np.sort(-np.reshape(data,nX*nY))  # Sort data descending

    # Downsampling of depth scan
    if N is None:
        N = nX*nY
    elif N != nX*nY:
        f = interpolate.interp1d(np.arange(0,nX*nY),depth_scan)
        x2 = np.arange(0,N)*nX*nY/N
        depth_scan = f(x2)

    # Fill d-scan matrix
    if d0 is None:
        # Create first column
        d0 = depth_scan.reshape(1,N)
    else:
        # Add new column to existing one
        d0 = np.append(d0,depth_scan.reshape(1,N),axis=0)

# Cropping of negative data causes there are many redundant zeros in `d0` ->
# -> get index of last non-zero element and crop `d0` to new size
d0N = np.max(np.nonzero(d0))+1
d0 = d0[:,0:d0N]

# Data can be optionaly edited when some strange features appear.
# Hey, these data are real, taken by WLI microscope so there can be some strange
# things, right?
d0[-1,0] = d0[-1,1]
d0[-2,0:1] = d0[-2,2]

# Area vector of D-scan
S0 = np.arange(0,d0N)*(width*height/N)

# Plot depth scan
fig = plt.figure()
plt.title('Depth scan')
for i in range(len(d0)):
    plt.plot(S0,d0[i],label=str(i))
plt.xlabel(r'S [mm$^2$]')
plt.ylabel('Depth [nm]')
plt.legend()
plt.show()

# %% ======================= Initialization of matrices ========================

# Dimensions of matrix (`nD` and `nS`) are experimentally quessed. Depth scans
# (i.e. curves with constant energy) should be continuous for best results.
# Large matrix does not ensures convergence.
# Too small matrix does not have enough resolution.

# Size in depth-direction, step, axis vector
nD = 200; dD = np.max(d0)/(nD-1); d_axis = np.arange(0,nD)*dD
# Size in area-direction, step, axis vector
nS = 200; dS = np.max(S0)/(nS-1); s_axis = np.arange(0,nS)*dS

# Depth scan matrix (`DS`): It is identical to the depth scan but shaped as
# matrix. Values of the pixels are pulse energies corresponding to the given
# imprint.
DS = np.zeros((nD,nS))
# Inverse depth scan matrix (`iDS`): Same as `DS` but values equal to 1/energy.
iDS = np.zeros((nD,nS))

# Fill matrices with numbers
for i in range(len(d0)):
    print(f"Reading scan {i+1}/{len(d0)}")
    for d in range(d0N):
        id = np.round(d0[i,d]/dD).astype(int)   # index of depth
        iS = np.round(S0[d]/dS).astype(int)     # index of area
        if id>=0 and iS>=0:
            DS[id,iS]  = energies[i]
            iDS[id,iS] = 1/energies[i]

# `DS` matrix may contain gaps in columns. Following procedure removes these
# gaps. This is realy important for large matrices.
for iS in range(nS):
    for id in range(nD):
        if DS[id,iS]>0:
            e = DS[id,iS]
            for id2 in range(id,nD):
                if DS[id2,iS] == e:
                    for id3 in range(id,id2):
                        DS[id3,iS]  = e
                        iDS[id3,iS] = 1/e

fig = plt.figure()
plt.imshow(DS,origin='lower',aspect='auto',interpolation='none',
    extent =[s_axis.min(), s_axis.max(), d_axis.min(), d_axis.max()])
plt.title('DS matrix')
plt.xlabel(r'S [mm$^2$]')
plt.ylabel('Depth [nm]')
plt.show()


# %% ===================== Init f-scan and dose vector =========================

DS_count = (DS>0)

E0 = (DS[:,0]).copy()       # just to keep original value of E
E = E0.copy()               # E is a dose vector
E_count = (E>0)             # nonzero E-counter
E_avg = E.copy()            # averaged dose vector
E_avg_count = 1*(E_avg>0)   # counter of non-negative values of E_avg

f = np.zeros(nS)            # fluence scan vector
f_count = f.copy()          # counter of f-scan
f_avg = f.copy()            # averaged f vector
f_avg_count = (f_avg>0)     # counter of non-negative values of f_avg


# %% =========================== Iteraction loop ===============================

# Convergence is usually quite fast, 100 of steps should be enough.
no_iterations = 100
for i in range(no_iterations):

    df = E_avg.dot(iDS)
    f = f + df
    f_count = f_count + E_avg_count.dot(DS_count)
    f_avg[f_count>0] = f[f_count>0] / f_count[f_count>0]

    f_avg_count = 1*(f_avg>0)

    f_avg = (f_avg-np.min(f_avg))/(np.max(f_avg)-np.min(f_avg))
    f_avg[0] = 1

    dE = DS.dot(f_avg)
    E = E + dE
    E_count = E_count + DS_count.dot(f_avg_count)
    E_avg[E_count>0] = E[E_count>0] / E_count[E_count>0]

    E_avg_count = 1*(E_avg>0)

# %% ========================= Plot final figures ==============================

fig = plt.figure()
plt.title('Calibration curve')
plt.plot(d_axis,E0,label=r'Initial E$_0$')
plt.plot(d_axis,E_avg,label=r'$E$ after '+str(no_iterations)+' iterations')
plt.xlabel('Depth [nm]')
plt.ylabel(r'$E$ [mJ]')
plt.legend()
plt.show()

fig = plt.figure()
plt.title('Fluence scan')
plt.plot(s_axis,f_avg)
plt.xlabel(r'$S$ [mm$^2$]')
plt.ylabel('Fluence [1]')
plt.show()

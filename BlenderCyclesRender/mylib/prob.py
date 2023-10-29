import numpy as np

def generatePointDist(L, dr, shift_xyz, mu, sigma, Nc, A):
    N = int(L/dr) #number of steps
    theta_start = A[0]
    theta_end = A[1]
    theta = theta_end-theta_start
    phi_start = A[2]
    phi_end = A[3]
    phi = phi_start + phi_end

    Sum = 0
    for i in range(N):
        ri = i * dr
        Sum = cal_pdf(ri,mu,sigma) + Sum

    n = np.zeros(N)
    for i in range(N):
        ri = i * dr
        n[i] = int(cal_ni(ri,mu,sigma,Nc,Sum))

    ri = []
    thetai = []
    phii = []
    for i in range(N):
        ri = np.append(ri,(np.random.random(int(n[i]))*dr + i*dr)) 
        thetai = np.append(thetai,(np.random.random(int(n[i]))*theta + theta_start))
        phii = np.append(phii,(np.pi/2 - np.random.random(int(n[i]))*phi + phi_start))
    #ri = ri + shift_from_center
    #Cartesian coordinates
    x = ri * np.sin(phii) * np.cos(thetai)
    y = -ri * np.sin(phii) * np.sin(thetai)
    z = ri * np.cos(phii) + 0.1
    
    RCL = np.zeros((3,z.shape[0]))
    RCL[0,:] = x + shift_xyz[0]
    RCL[1,:] = y + shift_xyz[1]
    RCL[2,:] = z + shift_xyz[2]
    
    return RCL

def cal_pdf(ri,mu,sigma): #probability density function
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-1/2)*((ri-mu)/sigma)**2)

def cal_ni(ri,mu,sigma,Nc, Sum):
    return cal_pdf(ri,mu,sigma)*Nc/Sum
    
def randomCamLoc(ri,dr,shift_from_center,mu,sigma,Nc,N,A):
    theta_start = A[0]
    theta_end = A[1]
    theta = A[2]
    phi_start = A[3]
    phi_end = A[4]
    phi = A[5]
    
    Sum = 0
    for i in range(N):
        ri = i * dr
        Sum = cal_pdf(ri,mu,sigma) + Sum

    n = np.zeros(N)
    for i in range(N):
        ri = i * dr
        n[i] = int(cal_ni(ri,mu,sigma,Nc,Sum))

    ri = []
    thetai = []
    phii = []
    for i in range(N):
        ri = np.append(ri,(np.random.random(int(n[i]))*dr + i*dr)) 
        thetai = np.append(thetai,(np.random.random(int(n[i]))*theta + theta_start))
        phii = np.append(phii,(np.pi/2 - np.random.random(int(n[i]))*phi + phi_start))
    #ri = ri + shift_from_center
    #Cartesian coordinates
    x = ri * np.sin(phii) * np.cos(thetai)
    y = -ri * np.sin(phii) * np.sin(thetai)
    z = ri * np.cos(phii) + 0.1
    
    RCL = np.zeros((3,z.shape[0]))
    RCL[0,:] = x
    RCL[1,:] = y - shift_from_center
    RCL[2,:] = z
    
    return RCL
    
def chooseRanPosVal(Max_Pos_val = 1, No_divisions=1000, mu = 0.5, sigma = 10, offset = 0.0001, Invert = True ):
    Sum = 0
    Y = []
    for xi in range(No_divisions):
        pdf = cal_pdf(xi,mu*No_divisions,sigma*No_divisions)
        Y = np.append(Y,pdf)
        Sum += pdf
    Y = Y - np.min(Y) + offset
    Y = Y*Max_Pos_val/np.max(Y)
    if Invert:
        Y = Max_Pos_val - Y
    np.random.shuffle(Y)
    return Y[0]

def getRanPosVal_around_point(point = 1, noise_range = 1, No_divisions=1000, mu = 0.2,sigma = 10, offset = 0, Invert = False ):
    # generate negative and positive position values
    Sum = 0
    Y = []
    for xi in range(0,int(No_divisions/2)):
        pdf = cal_pdf(xi,mu*No_divisions,sigma*No_divisions)
        Y = np.append(Y,pdf)
        Sum += pdf
    Y = Y - np.min(Y) + offset
    Y = Y*noise_range/np.max(Y)
    if Invert:
        Y = noise_range - Y
    Y1 = Y    
    Sum = 0
    Y = []
    for xi in range(-int(No_divisions/2),0):
        pdf = cal_pdf(xi,mu*No_divisions,sigma*No_divisions)
        Y = np.append(Y,pdf)
        Sum += pdf
    Y = Y - np.min(Y) + offset
    Y = Y*noise_range/np.max(Y)
    if Invert:
        Y = noise_range - Y
    Y2 = Y
    Y = np.concatenate((-Y2,Y1),axis=0)+point
    np.random.shuffle(Y)

    return Y[0]

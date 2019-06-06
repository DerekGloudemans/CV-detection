import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

# 1D trivial example
if False:
    obs_sigma = 4.5
    times = np.array([t/10.0 for t in range(0,100)])
    data =  times + 4
    measurements = np.array([item + random.gauss(0,obs_sigma) for item in data])
    dt = 1/10.0
    
    # state variable will contain p and v
    state_est = np.array([[0],[2]])
    
    ## make state covariance matrix
    sigma = np.array([1,1]) # errors in state, not sure what these correspond to
    cov = np.zeros([2,2])
    for i in range(0,len(cov)):
        for j in range(0,len(cov[0])):
            cov[i,j] = sigma[i]*sigma[j]
    
    ## make measurement covariance matrix
    R = np.array([1])
    R = R[:,np.newaxis]
    
    # state transition matrix
    F = np.array([[1,dt],[0,1]])
    
    # Measurement transition matrix (i.e. how the current state is expected to produce measurements)
    H = np.array([1, dt])
    H = H[np.newaxis,:]
    
    # initialize results structure
    estimates = []
    measurement_only = []
    model_only = []
    for i in range(0,len(measurements)):
        #append current state_estimate to list
        estimates.append(state_est[0])
        
        # Compute state estimate x` = Fx + Bu (none)
        state_est = np.matmul(F,state_est)
        
        # Covariance update P` = FPF'
        cov_est = np.matmul(np.matmul(F,cov),np.transpose(F))
    
        # measurement error y = z-Hx`
        info = measurements[i] - np.matmul(H,state_est)
        
        # update sensor covariance S = HP`H' + R
        S = np.matmul(np.matmul(H,cov_est),np.transpose(H)) + R
        
        # calculate Kalman gain K = P`H'(S^-1)
        K = np.matmul(np.matmul(cov_est,np.transpose(H)),np.linalg.inv(S))
    
        # get stats for model and measurement
        measurement_only.append((state_est + np.matmul(np.array([[1],[1]]),info))[0])
    
        # get new estimate
        state_est = state_est + np.matmul(K,info)
        
        
        # get new covariance mat
        cov = np.matmul((np.identity(2) - np.matmul(K,H)),cov_est)
        
        
    # plot results
    plt.title("Simple 1D Kalman Filtering Example")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.plot(times,data)
    plt.plot(times,measurements,'.')
    plt.plot(times,estimates,'-')
    plt.legend(['Real Data', 'Observations', 'Estimate','Model'])
    
 ###########################################################################   
# n = 10 state size example
 
sampling_error = 100
model_err = 1
meas_err= 100
t = 1/30.   # discrete time step 
 
x = np.zeros([10,1])          # intialize state
Q = np.zeros([10,10])
Q[[2,5,7,9],[2,5,7,9]] = model_err  # intialize model error covariance matrix
R = np.identity(4)* meas_err      # initialize measurement error covariance matrix

F = np.identity(10)                # initialize state transition matrix
for i in range(len(F)-1):
    F[i,i+1] = t
    
H = np.zeros([4,10])               # initialize measurement transition matrix
H[[0,1,2,3],[0,3,6,8]] = 1

P = np.zeros([10,10])              # initialize state error covariance matrix


#generate data
time = [i/30.0 for i in range(0,1000)]
y_pos = [14*i * 10.5 + random.gauss(0,2) for i in time]
x_pos = [(10*i**2 * 5.5 + i) + random.gauss(0,1) for i in time]
scale = [100/(2*i+1) for i in time]
scale.reverse()
ratio = [1+i/3 for i in time]


kal_x = []
kal_y = []
kal_scale = []
kal_ratio = []

meas_x = []
meas_y = []
meas_scale = []
meas_ratio = []

# Kalman filtering loop here
for i in range(0,len(time)-1): # since wont have a measurement for the last state
    
    x_hat = np.matmul(F,x) #predicted a prior state estimate
    P = np.matmul(np.matmul(F,P),np.transpose(F)) + Q # predicted a priori state error covariance
    z = np.array([[x_pos[i+1]],[y_pos[i+1]],[scale[i+1]],[ratio[i+1]]]) # measurement at time i + 1
    z = z+(np.random.normal(0,sampling_error,(4,1))) # add some noise to measurements
    y = z - np.matmul(H,x_hat) # innovation 
    S = R + np.matmul(np.matmul(H,P),np.transpose(H)) # innovation covariance
    K = np.matmul(np.matmul(P,np.transpose(H)),np.linalg.inv(S)) # optimal kalman gain
    x = x_hat + np.matmul(K,y) # updated a posteriori estimate of state
    P = np.matmul((np.identity(10) - np.matmul(K,H)),P) # updated a posteriori estimate of state error covarinace
    
    # append to results
    kal_x.append(x[0,0])
    kal_y.append(x[3,0])
    kal_scale.append(x[6,0])
    kal_ratio.append(x[8,0])
    
    meas_x.append(z[0,0])
    meas_y.append(z[1,0])
    meas_scale.append(z[2,0])
    meas_ratio.append(z[3,0])
    

def plot_rectangles(im,x_pos,y_pos,scale,ratio,time,plot_pos = 0):
    for i in range(0,len(time)):
        if False:
            if plot_pos == 0: # changing colors
                color = (170,50+int(time[i]*50),20)
            elif plot_pos == 1:
                color = (20,170,50+int(time[i]*50))
            else:
                color = (50+int(time[i]*50),20,170)
        if plot_pos == 2:
            color = (100,100,0)
        elif plot_pos == 1:
            color = (240,240,240)
        else:
            color = (0,0,0)
        cv2.rectangle(im,(int(x_pos[i]-scale[i]/2),int(y_pos[i]-scale[i]*ratio[i]/2)),\
                      (int(x_pos[i]+scale[i]/2),int(y_pos[i]+scale[i]*ratio[i]/2)),\
                      color,\
                      1)
    return im


if False:
    # plot rectangles
    height = 1000
    width = 2000
    im = np.zeros((height,width,3), np.uint8)+255
    
    
    im = plot_rectangles(im,x_pos[:-1],y_pos[:-1],scale[:-1],ratio[:-1],time[:-1],plot_pos = 0)
    im = plot_rectangles(im,meas_x,meas_y,meas_scale,meas_ratio,time[:-1],plot_pos = 1)
    im = plot_rectangles(im,kal_x,kal_y,kal_scale,kal_ratio,time[:-1],plot_pos = 2)
    
    
    cv2.imshow('frame',im)
    while 1:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

plt.figure()
plt.plot(x_pos,y_pos)
plt.plot(meas_x,meas_y,'.')
plt.plot(kal_x,kal_y)
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("4D Kalman filtering Example")
plt.legend(["Ground Truth", "Measurements", "Kalman Filter"])


#Observability test
#import numpy as np
#
#t = .25
#F = np.zeros([12,12])
#
#for i in range(0,4):
#    F[3*i,3*i] = 1 # set position = 1
#    F[3*i,3*i +1] = t
#    F[3*i+1,3*i+1] = 1
#    F[3*i+1,3*i+2] = t
#    F[3*i+2, 3*i+2] = 1
#    
#H = F[[0,3,6,9],:]
#
#Q = H
#
#for i in range(1,12):
#    F_exp = np.identity(len(F))
#    for j in range(1,i):
#        F_exp = np.matmul(F_exp,F)
#    HF_exp = np.matmul(H,F_exp)
#    Q = np.concatenate((Q,HF_exp),0)
#    
#rank = np.linalg.matrix_rank(Q)
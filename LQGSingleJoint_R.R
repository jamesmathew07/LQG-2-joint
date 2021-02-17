   
#Script LQG   

# Define the linear state-space representation of the system:
  
# x[k+1] = Ax[k] + Bu[k] + xi[k]

library(MASS)
rm(list = ls());
dev.off();

#single joint reaching movements
G=0.14        #viscous constant Ns/m
I=0.1;        #Inertia kgm2
tau = 0.066;  # muscle time constant s

A = t(matrix(c(0,1,0,0,-G/I,1/I, 0,0,-1/tau), ncol=3))
B = matrix(c(0,0,1/tau),ncol=1)

ns = nrow(A); #dimenstion of state
nc = ncol(B); #dimension of control
# transformation into discrete time system
delta = 0.01; #time step
A = diag(ns)+delta*A
B= delta*B;

#augment the system to include a target state
A = rbind(cbind(A,matrix(0,ns,ns)),cbind(matrix(0,ns,ns),diag(ns)))
B = rbind(B,matrix(0,ns,nc))

#define cost function
rtime = 0.5;
nStep = rtime/delta; # 50 steps

Q = array(0,dim =c(2*ns,2*ns,nStep+1))
R = 10^-5*array(1,dim=c(nc,nc,nStep))

# fill in the cost of the last target
In = diag(ns)
w = c(1,1,0);
v= 1:ns
for (i in v){
ei = matrix(c(In[,i],-In[,i]),ncol=1) # how to do via-point ? target jump ? target tracking ?perturbations?
Q[,,dim(Q)[3]] = Q[,,dim(Q)[3]]+w[i]*(ei%*%t(ei))
}

#Backward recurrence for optimal feedback gains
S=Q;
oXi = 0.2*(B%*%t(B)) # noise covariance matrix
L= array(0,dim=c(nc,2*ns,nStep))
s=0;
v = c(nStep:1)
for (k in v){
  L[,,k] = (t(B)%*%S[,,k+1]%*%A)/c(R[,,k]+t(B)%*%S[,,k+1]%*%B);
  S[,,k] = Q[,,k]+(t(A)%*%S[,,k+1]%*%(A-B%*%L[,,k]))
  s = s+ sum(diag(S[,,k+1]+oXi))
}
gain = matrix(0,2,nStep);


v = 1:nStep
for (k in v){
  gain[1,k] = L[1,1,k];
  gain[2,k] = L[1,2,k];
}
#normalise gains
gain2 = matrix(0,2,nStep);
gain2[1,] = gain[1,]/max(gain[1,]);
gain2[2,] = gain[2,]/max(gain[2,]);

#plot control gain 
par(mfrow=c(1,3))
x1 = c(0.01:0.01:nStep*0.01)
y1 = gain2 
par(mfg=c(1,1));plot(x1,y1[1,],type="l",xlab="Time[s]", ylab="Normalized gain[a.u]");par(new=TRUE);
plot(x1,y1[2,],col="red",type="l");
title('Control gain');
legend("topleft",   legend = c("Angle","Velocity")) 

#forward recurrence of optimal kalman gains
H = diag(2*ns)                         # all state variables are measures independently
ny = nrow(H);
oOmega = 0.5*max(max(oXi))*diag(2*ns); # diagonal covariance matrix
Sigma = oOmega;                        # default initialization
K= array(0,dim =c(2*ns,ny,nStep))

v =1:nStep
for(k in v){

  K[,,k] = (A%*%Sigma%*%t(H))%*% solve(H%*%Sigma%*%t(H)+oOmega);
  Sigma = oXi + (A-K[,,k]%*%H)%*%Sigma%*%t(A);
}

#Simulations
nsimu =10 # perform 10 simulation runs
x = array(0,dim =c(2*ns,nStep+1,nsimu))   # initialize state
xhat = x;                                 # initialise state estimate
control = array(0,dim=c(nc,nStep,nsimu))  # initialise control
avControl = array(0,dim=c(nc,nStep));     # average control variable

ns2 = c(1:nsimu)
for(p in ns2)
  {
    x[(ns+1):(dim(x)[1]),1,p] = matrix(c(20*pi/180,0,0),ncol=1) # include the target

for(k in v){
  motorNoise    = (mvrnorm(n = 1, matrix(0,2*ns,1), oXi,  empirical = FALSE))       # motor noise
  sensoryNoise  = (mvrnorm(n = 1, matrix(0,2*ns,1), oOmega,  empirical = FALSE))  # sensory noise
  u             = -L[,,k]%*%x[,k,p];                                                   # control variable
  y             = H%*%x[,k,p] + sensoryNoise;                                          # state measurement
  xhat[,k+1,p]  = A%*%xhat[,k,p] +B%*%u + K[,,k]%*%(y- ((H%*%xhat[,k,p])))  # state estimate
  x[,k+1,p]     = A%*%x[,k,p]+ B%*%u + motorNoise;                             # dynamics
  control[,k,p] = u;
}

# Fill in the average control matrix
avControl = avControl + control[,,p]/nsimu;

y1= x[1,,p]*180/pi;
par(mfg=c(1,2)) ;par(new=TRUE);
plot(c(0.01:0.01:(nStep+1)),y1,type="l",xlab="Time[s]", ylab="Joint angle[deg]");
title('Trajectories');
par(mfg=c(1,3)) ;par(new=TRUE);
plot(c(0.01:0.01:(nStep)*0.01), control[1,,p],type="l",xaxt='n',ann=FALSE,axes=FALSE,ylim=c(-2,2))
}

#par(mfg=c(1,2)) ;par(new=TRUE);
#plot(xlab="Time[s]", ylab="Joint angle[deg]");
#title('Trajectories');

par(mfg=c(1,3)) ;par(new=TRUE);
plot(c(0.01:0.01:(nStep)*0.01), avControl[1,],type="l",xlab="Time[s]", ylab="Control[Nm]", col ="red", ylim=c(-2,2),lwd =2);
title('Control vector');

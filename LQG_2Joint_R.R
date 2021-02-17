
# Script LQG - 2 joint hand reach model

# import libraries
library(MASS)
rm(list = ls());
dev.off();
source("C:/Users/jmathew/Dropbox (INMACOSY)/James-UCL/LQG/OFG.R")

load = c(0,13,-13) # possible mechanical perturbation loads
for(Lo in load)
{
  gamma  = 0.5; # learning rate
  rtime  = 0.6; # 600 ms reach time 
  xinit  = matrix(c(0,0,0,0,0,0),ncol=1)
  xfinal = matrix(c(0,0.16,0,0,0,0),ncol=1);
  x0     = rbind(xinit,xfinal);
  Gx     = 0.1;
  Gy     = Gx;
  m      = 2.5;
  tau    = 0.1;
  lamda  = 0
  k      = 0
  A = t(matrix(c(0,0,1,0,0,0,
                 +     0,0,0,1,0,0, 
                 +   -k/m,0, -Gx/m, Lo/m, (m^-1),0,  
                 +     0, -k/m,0,-Gy/m,0,(m^-1),  
                 +     0,0,0,0,(-tau^-1),lamda/tau , 
                 +     0,0,0,0,lamda/tau,(-tau^-1)), ncol = 6))
  B = t(matrix(c(0, 0,
                 + 0, 0,
                 + 0, 0,
                 + 0, 0,
                 + tau^-1, lamda/tau,
                 + lamda/tau, tau^-1), nrow =2));
  
  ns = nrow(A); #dimenstion of state
  nc = ncol(B); #dimension of control
  
  # transformation into discrete time system
  delta = 0.01; #time step
  A     = diag(ns)+delta*A
  B     = delta*B;
  
  #augment the system to include a target state
  A         = rbind(cbind(A,matrix(0,ns,ns)),
                    +  cbind(matrix(0,ns,ns),diag(ns)))
  B         = rbind(B,matrix(0,ns,nc))
  Aest      = A;
  Aest[3,4] = 0; # Estimation without any perturbation
  
  #define cost function
  nStep = rtime/delta; # 60 steps
  Q     = array(0,dim =c(2*ns,2*ns,nStep+1))
  R     = array(0,dim =c(nc,nc,nStep))
  #R    = 10^-5*array(1,dim=c(nc,nc,nStep))
  for (k in c(1:nStep))
  {
    R[,,k] = (10^-5)*diag(nc);
  }
  
  # fill in the cost of the last target
  In = diag(ns)
  w  = c(1000,1000,20,20,0,0); # weights for position and velocity accuracy at end point
  for (i in c(1:ns))
  {
    ei             = matrix(c(In[,i],-In[,i]),ncol=1) 
    Q[,,dim(Q)[3]] = Q[,,dim(Q)[3]]+w[i]*(ei%*%t(ei))
  }
  
  for (i in c(1:nStep))
  {
    Q[,,i] = (i/nStep)^25*Q[,,dim(Q)[3]]
  }
  
  #Backward recurrence for optimal feedback gains
  L     = OFG(Aest,B,Q,R,x0,nStep)
  L2    = L;
  Q2    = Q;
  A2    = A;
  Aest2 = Aest;
  gain  = matrix(0,6,nStep);
  
  for (k in c(1:nStep))
  {
    gain[1,k] = L[1,1,k];
    gain[3,k] = L[1,3,k];
    gain[5,k] = L[1,5,k];
  }
  # normalise gains
  gain2 = matrix(0,6,nStep);
  gain2[1,] = gain[1,]/max(gain[1,]);
  gain2[3,] = gain[3,]/max(gain[3,]);
  gain2[5,] = gain[5,]/max(gain[5,]);
  
  #plot control gain 
  par(mfrow=c(1,4))
  x1 = c(0.01:0.01:nStep*0.01)
  y1 = gain2 
  par(mfg=c(1,1));plot(x1,y1[1,],type="l",ann=FALSE);par(new=TRUE);
  plot(x1,y1[3,],col="red",type="l",ann=FALSE);par(new=TRUE);
  plot(x1,y1[5,],col="green",type="l",ann=FALSE);
  title('Control gain',xlab="Time[s]", ylab="Normalized gain[a.u]");
  legend("topleft",legend = c("Pos","Vel","tor")) 
  
  #forward recurrence of optimal kalman gains
  H      = diag(2*ns) # all state variables are measures independently
  ny     = nrow(H);
  oXi    = (B%*%t(B)) # noise covariance matrix
  oOmega = 0.5*max(max(oXi))*diag(2*ns);  # diagonal covariance matrix
  Sigma  = oOmega;                        # default initialization
  K      = array(0,dim =c(2*ns,ny,nStep))
  
  for(k in c(1:nStep)){
    K[,,k] = (A%*%Sigma%*%t(H))%*% solve(H%*%Sigma%*%t(H)+oOmega);
    Sigma  = oXi + (A-K[,,k]%*%H)%*%Sigma%*%t(A);
  }
  
  #Simulations
  nsimu      = 10                                    # perform 10 simulation runs
  x          = array(0,dim =c(2*ns,nStep+1,nsimu))   # initialize state
  xhat       = x;                                    # initialise state estimate
  control1   = array(0,dim=c(nc,nStep,nsimu))        # initialise control
  control2   = array(0,dim=c(nc,nStep,nsimu))        # initialise control
  avControl1 = array(0,dim=c(nc,nStep)); 
  avControl2 = array(0,dim=c(nc,nStep));             # average control variable
  
  for(p in c(1:nsimu))
  {
    x[,1,p]    = rbind(xinit,xfinal);
    xhat[,1,p] = rbind(xinit,xfinal);
    L          = L2;
    Q          = Q2;
    A          = A2;
    Aest       = Aest2;
    
    for(k in c(1:(nStep-2)))
    {
      motorNoise    = (mvrnorm(n = 1, matrix(0,2*ns,1), oXi,  empirical = FALSE))       # motor noise
      sensoryNoise  = (mvrnorm(n = 1, matrix(0,2*ns,1), oOmega,  empirical = FALSE))    # sensory noise
      u             = -L[,,1]%*%x[,k,p];                                                # control variable
      L             = OFG(Aest,B,Q[,,(k+1):dim(Q)[3]],R[,,(k+1):dim(R)[3]],xhat[,k,p],nStep)
      
      x[,k+1,p]     = A%*%x[,k,p]+ B%*%u + motorNoise;  
      xhat[,k+1,p]  = Aest%*%x[,k,p] +B%*%u ;
      
      eps1      = x[1:ns,k+1,p]-xhat[1:ns,k+1,p];
      #Updating Model Matrices
      theta_t   = Aest[3,4];
      psy       = array(0, dim=c(1,ns));
      psy[1,3]  = x[4,k+1,p];
      theta_up  = theta_t + gamma[1]*(psy)%*%(eps1);
      Aest[3,4] = theta_up[1];
      
      control1[,k,p] = u[1];
      control2[,k,p] = u[2];
      
      par(mfg=c(1,2)) ;
      plot(k*0.01,x[1,k,p],type="p",col ="magenta",xaxt='n',yaxt='n',ann=FALSE,xlim=c(0,0.6001),ylim=c(-0.03,0.03));par(new=TRUE);
      
      par(mfg=c(1,3)) ;
      plot(k*0.01,control1[1,k,p],type="p", col ="magenta",xaxt='n',yaxt='n',ann=FALSE,xlim=c(0,0.6),ylim=c(-20,20));par(new=TRUE);
      par(mfg=c(1,3)) ;
      plot(k*0.01,control2[1,k,p],type="p", col ="green",xaxt='n',yaxt='n',ann=FALSE,xlim=c(0,0.6),ylim=c(-20,20));par(new=TRUE);
      par(mfg=c(1,4)) ;
      plot(x[1,,p],x[2,,p],type="p", col ="red",xaxt='n',yaxt='n',ann=FALSE,xlim=c(-0.03,0.03),ylim=c(0,0.2));par(new=TRUE);
      
    }
    
    #Fill in the average control matrix
    avControl1 = avControl1 + control1[,,p]/nsimu;
    avControl2 = avControl2 + control2[,,p]/nsimu;
    
    par(mfg=c(1,2)) ;par(new=TRUE);
    plot(c(0.01:0.01:(nStep+1)*0.01),x[1,,p],type="l",xlim=c(0,0.6001),ylim=c(-0.03,0.03),ann=FALSE,);par(new=TRUE);
    par(mfg=c(1,3)) ;par(new=TRUE);
    plot(c(0.01:0.01:(nStep)*0.01), control1[1,,p],type="l",xaxt='n',yaxt='n',xlim=c(0,0.6),ylim=c(-20,20),ann=FALSE);par(new=TRUE);
    par(mfg=c(1,3)) ;par(new=TRUE);
    plot(c(0.01:0.01:(nStep)*0.01), control2[1,,p],type="l",ann=FALSE,xlim=c(0,0.6),ylim=c(-20,20));par(new=TRUE);
    
    par(mfg=c(1,4)) ;par(new=TRUE);
    plot(x[1,,p],x[2,,p],type="l",ann=FALSE,xlim=c(-0.03,0.03),ylim=c(0,0.2));par(new=TRUE);
  }
}
par(mfg=c(1,2)) ;par(new=TRUE);
title(main= 'Trajectories',xlab="Time[s]", ylab="X position");

par(mfg=c(1,3)) ;par(new=TRUE);
plot(c(0.01:0.01:(nStep)*0.01), avControl1[1,],type="l",xaxt='n',yaxt='n',xlim=c(0,0.6),ylim=c(-20,20),ann=FALSE);
par(mfg=c(1,3)) ;par(new=TRUE);
plot(c(0.01:0.01:(nStep)*0.01), avControl2[1,],type="l",ann=FALSE,xlim=c(0,0.6),ylim=c(-20,20));
par(mfg=c(1,3)) ;par(new=TRUE);
title(main = 'Control vector',xlab="Time[s]", ylab="Control[Nm]");

par(mfg=c(1,4)) ;par(new=TRUE);
title(main ='X-Y position in workspace',xlab="X position", ylab="Y position");


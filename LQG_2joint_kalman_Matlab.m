%********************
%*    Script LQG  - 2 joint hand reaching model  *
%********************

clc;clear all;close all;

%-------------------------------------------------------------------------
% Lo   = 13; % load variable
for Lo = 15%[0 -13 13] % load variable
    gamma  = 0.001;
    rtime  = 0.6; %1000ms
    xinit  = [0 0 0 0 0 0]';
    xfinal = [0 .16 0 0 0 0]';
    x0     = [xinit;xfinal];
    
    Gx     = .1;
    Gy     = Gx;
    m      = 2.5;
    tau    = 0.1;
    lambda = 0;
    k      = 0;
    
    A = [0 0 1 0 0 0;
        0 0 0 1 0 0;
        -k/m 0 -Gx/m Lo/m m^-1 0;
        0 -k/m 0 -Gy/m 0 m^-1;
        0 0 0 0 -tau^-1 lambda/tau;
        0 0 0 0 lambda/tau -tau^-1];
    B = [0 0;
        0 0;
        0 0;
        0 0;
        tau^-1 lambda/tau;
        lambda/tau tau^-1];
        
    
    ns = size(A,1); % Dimension of the state
    nc = size(B,2); % Dimension of the control
    
    % Transformation into discrete time system: Choose a numerical integration
    delta = .01; % Time step
    A     = eye(ns) + delta*A;
    B     = delta*B;
    
    % Augment the system to include a target state:
    A = [A,zeros(ns,ns);
        zeros(ns,ns),eye(ns)];
    B = [B;
        zeros(ns,nc)];
    AestCont  = A;
    Aest      = AestCont;
    Aest(3,4) = 0;
    
    %-------------------------------------------------------------------------
    % Define the cost-function
    %
    % J(x,u) = sum x[k+1]Q[k+1]x[k+1] + u[k]R[k]u[k]
    
    nStep = rtime/delta;
    Q     = zeros(2*ns,2*ns,nStep+1);
    % R = 10^-5*ones(nc,nc,nStep); % exeption eye
    for i = 1:nStep
        R(:,:,i) = (10^-5)*eye(nc);
    end
    
    % Fill in the cost of the last target
    In = eye(ns);
    w  = [100000 100000 20 20 0 0];
    
    for i = 1:ns
        ei             = [In(:,i);-In(:,i)];
        Q(:,:,end)     = Q(:,:,end) + w(i)*(ei*ei');
    end
    
    for i = 1:nStep
        Q(:,:,i) = (i/(nStep))^25*Q(:,:,end);
    end
    
    %-------------------------------------------------------------------------
    % Backwards recurrence for the optimal feedback gains
    
    L     = OFG(Aest,B,Q,R,x0,nStep);
    L2    = L;
    Q2    = Q;
    A2    = A;
    Aest2 = Aest;
    gain  = zeros(6,nStep);
    
    for k = 1:nStep
        gain(1,k) = L(1,1,k);
        gain(3,k) = L(1,3,k);
        gain(5,k) = L(1,5,k);
    end
    
    subplot(161)
    plot([.01:.01:nStep*.01],gain./(max(gain,[],2)));
    xlabel('Time [s]'); ylabel('Gain, normalized [a.u.]');
    legend('Xpos','Xvel','Fx','Location','NorthWest'); title('Control Gains','FontSize',14);
    axis square
    
    %-------------------------------------------------------------------------
    % Forward recurrence for the optimal Kalman gains
    H      = eye(2*ns);      % All state variables are measured independently
    ny     = size(H,1);
    oXi    = (B*B');         % Noise covariance matrix
    oOmega = 0.5*max(max(oXi))*eye(2*ns);     % Diagonal covariance matrix
    Sigma  = oOmega;                          % Initialization (Default)
    K      = zeros(2*ns,ny,nStep);
    for k = 1:nStep
        K(:,:,k) = (A*Sigma*H')/(H*Sigma*H'+oOmega);
        Sigma = oXi + (A-K(:,:,k)*H)*Sigma*A';
    end
    
    gain2  = zeros(6,nStep);
    
    for k = 1:nStep
        gain2(1,k) = K(1,1,k);
        gain2(3,k) = K(1,3,k);
        gain2(5,k) = K(1,5,k);
    end
    
    subplot(166)
    plot([.01:.01:nStep*.01],gain2./(max(gain2,[],2)));hold on;
    xlabel('Time [s]'); ylabel('Kalman Gain, normalized [a.u.]');
    legend('Xpos','Xvel','Fx','Location','NorthWest'); title('Sensory Error Gains','FontSize',14);
    axis square
    %-------------------------------------------------------------------------
    % Simulations
    
    nsimu      = 10;                         % Performs 10 simulation runs
    x          = zeros(2*ns,nStep+1,nsimu);  % Initialize the state
    xhat       = x;                          % Initialize the state estiamte
    control1   = zeros(nc,nStep,nsimu);      % Initialize control
    control2   = zeros(nc,nStep,nsimu);
    sens1      = zeros(nc,nStep,nsimu);      % Initialize control
    sens2      = zeros(nc,nStep,nsimu);
    avControl1 = zeros(nc,nStep);            % Average Control variable
    avControl2 = zeros(nc,nStep);
    avSens1    = zeros(nc,nStep);            % Average Control variable
    avSens2    = zeros(nc,nStep);
    
    for p = 1:nsimu
        x(:,1,p)    = [xinit;xfinal];
        xhat(:,1,p) = [xinit;xfinal];
        L           = L2;
        Q           = Q2;
        A           = A2;
%         Aest        = Aest2;
        
        for k = 1:nStep-1
            motorNoise    = mvnrnd(zeros(2*ns,1),oXi)';        % motor noise
            sensoryNoise  = mvnrnd(zeros(2*ns,1),oOmega)';     % sensory noise
            u             = -L(:,:,1)*xhat(:,k,p);           % control variable
            [L]           = OFG(Aest,B,Q(:,:,k+1:end),R(:,:,k+1:nStep),xhat(:,k,p),nStep);
            y             = H*x(:,k,p) + sensoryNoise;   % state measurement
            yhat          = H*xhat(:,k,p);
            x(:,k+1,p)    = A*x(:,k,p) + B*u + motorNoise;     % dynamics
            [K]           = KAL(Aest,B,nStep);
            
            % 1 **********
            % no kalman no delay, fully observable (working, uncomment this and comment 2 & 3 below)
%             xhat(:,k+1,p) = Aest*x(:,k,p) + B*u ;
            % 1 **********
            
            % 2 **********
            % kalman without delay (working, uncomment this and comment 1 & 3 )
%             xhat(:,k+1,p) = Aest*x(:,k,p) + B*u + ...       
%                             K(:,:,k)*(y-H*xhat(:,k,p));  % State Estimate
            % 2 **********

            % 3 **********
            % kalman with delay (working, uncomment this and comment 1 & 2)
            d = 25 ; %*10 ms sensory feedback delay  (play with d=20:40 and see difference)
            if k>(d+1) % during iteration if time > delay time introduce sensory feedback in the estimate
            yd            = H*x(:,k-d,p) + sensoryNoise;   % (delayed) state measurement is taken as the sensory FB, take before d time point
            xhat(:,k+1,p) = Aest*x(:,k,p) + B*u + ...       
                        K(:,:,k-d)*(yd-H*xhat(:,k-d,p));  % State Estimate considering delayed sensory error (sensory error before d time point)
            else % no sensory feedback
            xhat(:,k+1,p) = Aest*x(:,k,p) + B*u ; % fully observable, no sensory FB
            end
            % 3 **********
            
            eps1      = x(1:ns,k+1,p)-xhat(1:ns,k+1,p);
            % Updating Model Matrices
            theta_t   = [Aest(3,4)]';
            psy       = zeros(1,ns);
            psy(1,3)  = xhat(4,k+1,p);
            theta_up  = theta_t + gamma(1)*psy*eps1;
            Aest(3,4) = theta_up(1);
            
            control1(:,k,p) = u(1);
            control2(:,k,p) = u(2);
            sens1(:,k,p) = yhat(1);
            sens2(:,k,p) = yhat(2);
            
            subplot(162); plot(k*.01,x(1,k,p),'.m'), hold on;
            subplot(163);
            plot(k*.01,control1(1,k,p),'.m'), hold on;
            plot(k*.01,control2(1,k,p),'.g'), hold on;
            subplot(164);   plot(x(1,:,p),x(2,:,p),'.r'), hold on;
            
            subplot(165);
            plot(k*.01,sens1(1,k,p),'m'), hold on;
            plot(k*.01,sens2(1,k,p),'g'), hold on;
        end
        
        % Fill in the average control matrix
        avControl1 = avControl1 + control1(:,:,p)/nsimu;
        avControl2 = avControl2 + control2(:,:,p)/nsimu;
        
        avSens1 = avSens1 + sens1(:,:,p)/nsimu;
        avSens2 = avSens2 + sens2(:,:,p)/nsimu;
        
        subplot(162)
        plot([.01:.01:(nStep+1)*.01],x(1,:,p)), hold on;
        
        subplot(163)
        plot([.01:.01:(nStep)*.01],control1(1,:,p)), hold on;
        plot([.01:.01:(nStep)*.01],control2(1,:,p)), hold on;
        
        subplot(164)
        plot(x(1,:,p),x(2,:,p),'.b'); hold on;
        subplot(165)
        plot([.01:.01:(nStep)*.01],sens1(1,:,p)), hold on;
        plot([.01:.01:(nStep)*.01],sens2(1,:,p)), hold on;
        
        for k = 1:nStep
        gain2(1,k) = K(1,1,k);
        gain2(3,k) = K(1,3,k);
        gain2(5,k) = K(1,5,k);
        end
    
    subplot(166)
    plot([.01:.01:nStep*.01],gain2./(max(gain2,[],2)));hold on;
%         pause
    end
end

subplot(162)
xlabel('Time [s]'); ylabel('X'); title('Trajectories','FontSize',14);
axis square

subplot(163)
plot([.01:.01:(nStep)*.01],avControl1(1,:),'k','Linewidth',1)
plot([.01:.01:(nStep)*.01],avControl2(1,:),'k','Linewidth',1)
xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector','FontSize',14);
axis square

subplot(164)
xlabel('X');ylabel('Y');title('Position','FontSize',14);
axis square

subplot(165)
plot([.01:.01:(nStep)*.01],avSens1(1,:),'k','Linewidth',1)
plot([.01:.01:(nStep)*.01],avSens2(1,:),'g','Linewidth',1)
xlabel('Time [s]'); ylabel('Sensory Signal [Nm]'); title('Sensory Vector','FontSize',14);
axis square

function [L] = OFG(A,B,Q,R,x0,nStep)
% Backwards recurrence for the optimal feedback gains
ns  = size(A,1); % Dimension of the state
nc  = size(B,2); % Dimension of the control
n   = size(Q,3);
St  = Q(:,:,end);     % initialize with the same dimension
oXi = (B*B');         % Noise covariance matrix
L   = zeros(nc,ns,nStep);
s   = 0;

for i = n-1:-1:1
    L(:,:,i) = (R(:,:,i)+B'*St*B)\B'*St*A;
    Sttemp = St;
    St = Q(:,:,i)+A'*Sttemp*(A-B*L(:,:,i));
    s = s+trace(Sttemp+oXi);
end
end
function [K] = KAL(A,B,nStep)
    % Forward recurrence for the optimal Kalman gains
    ns     = size(A,1); % Dimension of the state
    H      = eye(ns);      % All state variables are measured independently
    ny     = size(H,1);
    oXi    = (B*B');         % Noise covariance matrix
    oOmega = 0.5*max(max(oXi))*eye(ns);     % Diagonal covariance matrix
    Sigma  = oOmega;                          % Initialization (Default)
    K      = zeros(ns,ny,nStep);
    
    for k = 1:nStep
        K(:,:,k) = (A*Sigma*H')/(H*Sigma*H'+oOmega);
        Sigma = oXi + (A-K(:,:,k)*H)*Sigma*A';
    end
end

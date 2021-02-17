%********************
%*    Script LQG  - 2 joint hand reaching model  *
%********************

clc;clear all;close all;

%-------------------------------------------------------------------------
% Lo   = 13; % load variable
for Lo = [-15 15]% [0 -13 13] % load variable
    gamma  = 0.01;
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
        -k/m 0 (-Gx-Lo)/m Lo/m m^-1 0;
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
    w  = [1000 1000 20 20 0 0];
    
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
%     figure(1)
    subplot(141)
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
    
    %-------------------------------------------------------------------------
    % Simulations
    
    nsimu      = 11;                         % Performs 10 simulation runs
    x          = zeros(2*ns,nStep+1,nsimu);  % Initialize the state
    xhat       = x;                          % Initialize the state estiamte
    control1   = zeros(nc,nStep,nsimu);      % Initialize control
    control2   = zeros(nc,nStep,nsimu);
    avControl1 = zeros(nc,nStep);            % Average Control variable
    avControl2 = zeros(nc,nStep);
    
    for p = 1 :nsimu
        
        if p==2
        xfinal = [0.11 0.11 0 0 0 0]';
        A(3,4)    = 0;
        elseif p==4
        xfinal = [-0.11 0.11 0 0 0 0]';
        A(3,4)    = 0;
        elseif p==6
        xfinal = [0.16 0 0 0 0 0]';
        A(3,4)    = 0;
        elseif p==8
        xfinal = [-0.16 0 0 0 0 0]';
        A(3,4)    = 0;
        elseif p==10
        xfinal = [0 0.16 0 0 0 0]';
        A(3,4)    = 0;
        else
        xfinal = [0 0.16 0 0 0 0]';
        A(3,4)   = (Lo/m)*0.01;
        Aest     = Aest2;
        end
        
        
        
        x(:,1,p)    = [xinit;xfinal];
        xhat(:,1,p) = [xinit;xfinal];
%         L           = L2;
%         Q           = Q2;
%         A           = A2;
%         Aest        = Aest2;

        for k = 1:nStep-1
            
            motorNoise   = mvnrnd(zeros(2*ns,1),oXi)';        % motor noise
            %             sensoryNoise = mvnrnd(zeros(2*ns,1),oOmega)';     % sensory noise
            
            u   = -L(:,:,1)*x(:,k,p);           % control variable
            [L] = OFG(Aest,B,Q(:,:,k+1:end),R(:,:,k+1:nStep),xhat(:,k,p),nStep);
            
            %         y = H*x(:,k,p) + sensoryNoise;   % state measurement
            
            x(:,k+1,p)    = A*x(:,k,p) + B*u + motorNoise;     % dynamics
            xhat(:,k+1,p) = Aest*x(:,k,p) + B*u ;%+...        % State Estimate
            %             K(:,:,k)*(y-H*xhat(:,k,p));
            
            eps1      = x(1:ns,k+1,p)-xhat(1:ns,k+1,p);
            % Updating Model Matrices
            theta_t   = [Aest(3,4)]';
            psy       = zeros(1,ns);
            psy(1,3)  = x(4,k+1,p);
            theta_up  = theta_t + gamma(1)*psy*eps1;
            Aest(3,4) = theta_up(1);
            
            control1(:,k,p) = u(1);
            control2(:,k,p) = u(2);
%             figure(1)
            subplot(142); plot(k*.01,x(1,k,p),'.m'), hold on;
            subplot(143);
            plot(k*.01,control1(1,k,p),'.m'), hold on;
            plot(k*.01,control2(1,k,p),'.g'), hold on;
            subplot(144);   plot(x(1,:,p),x(2,:,p),'.r'), hold on;
            
            
        end
%         figure(1)
        % Fill in the average control matrix
        avControl1 = avControl1 + control1(:,:,p)/nsimu;
        avControl2 = avControl2 + control2(:,:,p)/nsimu;
        
        subplot(142)
        plot([.01:.01:(nStep+1)*.01],x(1,:,p)), hold on;
        
        subplot(143)
        plot([.01:.01:(nStep)*.01],control1(1,:,p)), hold on;
        plot([.01:.01:(nStep)*.01],control2(1,:,p)), hold on;
        
        subplot(144)
        plot(x(1,:,p),x(2,:,p)); hold on;
        figure(2)
        plot(x(1,:,p),x(2,:,p)); hold on;
%         pause
    end
end
% figure(1)
subplot(142)
xlabel('Time [s]'); ylabel('X'); title('Trajectories','FontSize',14);
axis square

subplot(143)
plot([.01:.01:(nStep)*.01],avControl1(1,:),'k','Linewidth',1)
plot([.01:.01:(nStep)*.01],avControl2(1,:),'k','Linewidth',1)
xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector','FontSize',14);
axis square

subplot(144)
xlabel('X');ylabel('Y');title('Position','FontSize',14);
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

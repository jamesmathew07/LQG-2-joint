%********************
%*    Script LQG    *
%********************


% Define the linear state-space representation of the system:
%
% x[k+1] = Ax[k] + Bu[k] + xi[k]
%
% First question: what is the continuous time differential equation?
% or equivalently, what is the biomechanical model.


%-------------------------------------------------------------------------
% Single joint reaching movements:
G = .14;        % Viscous Constant: Ns/m
I = .1;         % Inertia Kgm2
tau = 0.066;    % Muscle time constant, s

A = [0 1 0;0 -G/I 1/I;0 0 -1/tau];
B = [0;0;1/tau];

% A = [0 0 1 0;0 0 0 1;0 0 0 0;0 0 0 0];
% B = [0 0;0 0;1 0;0 1];

ns = size(A,1); % Dimension of the state
nc = size(B,2); % Dimension of the control

% Transformation into discrete time system: Choose a numerical integration
delta = .01; % Time step
A = eye(ns) + delta*A;
B = delta*B;

% Augment the system to include a target state:
A = [A,zeros(ns,ns);zeros(ns,ns),eye(ns)];
B = [B;zeros(ns,nc)];

%-------------------------------------------------------------------------
% Define the cost-function
%
% J(x,u) = sum x[k+1]Q[k+1]x[k+1] + u[k]R[k]u[k]

rtime = 0.6;
nStep = rtime/delta;

Q = zeros(2*ns,2*ns,nStep+1);
R = 10^-5*ones(nc,nc,nStep);

% Fill in the cost of the last target
In = eye(ns);
w = [1 1 0];

for i = 1:ns
    
    ei = [In(:,i);-In(:,i)];                         % How to do via-point? Target Jump? Target Tracking ? Perturbations?
    Q(:,:,end) = Q(:,:,end) + w(i)*(ei*ei');
    
end


%-------------------------------------------------------------------------
% Backwards recurrence for the optimal feedback gains

S = Q;                    % initialize with the same dimension
oXi = 0.1*(B*B');         % Noise covariance matrix 
L = zeros(nc,2*ns,nStep);
s = 0;

for k = nStep:-1:1
    
    L(:,:,k) = (R(:,:,k)+B'*S(:,:,k+1)*B)\B'*S(:,:,k+1)*A;
    S(:,:,k) = Q(:,:,k) + A'*S(:,:,k+1)*(A-B*L(:,:,k));
    s = s + trace(S(:,:,k+1)+oXi);
    
end

gain = zeros(2,nStep);

for k = 1:nStep
    
    gain(1,k) = L(1,1,k);
    gain(2,k) = L(1,2,k);
    
end

h1 = figure;
figure(h1);
subplot(131)
plot([.01:.01:nStep*.01],diag(max(gain,[],2))\gain);
xlabel('Time [s]'); ylabel('Gain, normalized [a.u.]');
legend('Angle','Velocity','Location','NorthWest'); title('Control Gains','FontSize',14);
axis square
% pause


%-------------------------------------------------------------------------
% Forward recurrence for the optimal Kalman gains

H = eye(2*ns);                        % All state variables are measured independently
ny = size(H,1);
oOmega = .1*max(max(oXi))*eye(2*ns);  % Diagonal covariance matrix
Sigma = oOmega;                       % Initialization (Default)    
K = zeros(2*ns,ny,nStep);

for k = 1:nStep
    
    K(:,:,k) = A*Sigma*H'/(H*Sigma*H'+oOmega);
    Sigma = oXi + (A-K(:,:,k)*H)*Sigma*A';
    
end
   
%-------------------------------------------------------------------------
% Simulations 

nsimu = 10;                         % Performs 10 simulation runs
x = zeros(2*ns,nStep+1,nsimu);      % Initialize the state
xhat = x;                           % Initialize the state estiamte
control = zeros(nc,nStep,nsimu);    % Initialize control
avControl = zeros(nc,nStep);        % Average Control variable


for p = 1:nsimu
    
    x(ns+1:end,1,p) = [20*pi/180;0;0];    % include the target
    xhat(ns+1:end,1,p) = [20*pi/180;0;0];    
    
    for k = 1:nStep
        
        motorNoise = mvnrnd(zeros(2*ns,1),oXi)';        % motor noise
        sensoryNoise = mvnrnd(zeros(2*ns,1),oOmega)';   % sensory noise
        
        u = -L(:,:,k)*xhat(:,k,p);                      % control variable
        
        %u = -L(:,:,k)*x(:,k,p);                      % control variable
        %fully observable
        
        y = H*x(:,k,p) + sensoryNoise;                  % state measurement
        
        xhat(:,k+1,p) = A*xhat(:,k,p) + B*u +...        % State Estimate
            K(:,:,k)*(y-H*xhat(:,k,p));
        
        x(:,k+1,p) = A*x(:,k,p) + B*u + motorNoise;     % dynamics
        control(:,k,p) = u;
        
    end
    
    % Fill in the average control matrix
    avControl = avControl + control(:,:,p)/nsimu;
    
    subplot(132)
    plot([.01:.01:(nStep+1)*.01],x(1,:,p)*180/pi), hold on;
    
    subplot(133)
    plot([.01:.01:(nStep)*.01],control(1,:,p)), hold on;
    
end

subplot(132)
xlabel('Time [s]'); ylabel('Joint angle [deg]'); title('Trajectories','FontSize',14);
axis square

subplot(133)
plot([.01:.01:(nStep)*.01],avControl(1,:),'k','Linewidth',2)
xlabel('Time [s]'); ylabel('Control [Nm]'); title('Control Vector','FontSize',14);
axis square




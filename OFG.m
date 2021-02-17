function [L] = OFG(A,B,Q,R,x0,nStep)
% Backwards recurrence for the optimal feedback gains
ns = size(A,1); % Dimension of the state
nc = size(B,2); % Dimension of the control

n   = size(Q,3);
St  = Q(:,:,end);      % initialize with the same dimension
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
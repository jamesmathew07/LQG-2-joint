OFG = function(A1, B1, Q1, R1, x1,nStep1)
{
  ns1   = nrow(A1)
  nc1   = ncol(B1)
  n1    = dim(Q1)[3]
  S1    = Q1[, , dim(Q1)[3]]
  oXi1  = (B1 %*% t(B1)) # noise covariance matrix
  L1    = array(0, dim = c(nc1, ns1, nStep1))
  s1    = 0

  if(length(dim(R1))<3)
    {
    R2      = array(0, dim = c(nc1,nc1,1))
    R2[,,1] = R1;
    }
  else 
    {
    R2 = R1;
    }

  for (k1 in c((n1-1):1)) 
 {
    L1[, , k1] = solve((R2[,,k1] + t(B1) %*% S1 %*% B1) , (t(B1) %*% S1 %*% A1))
    St1        = S1
    S1         = Q1[, , k1] + (t(A1) %*% St1 %*% (A1 - (B1 %*% L1[, , k1])))
    s1         = s1 + sum(diag(St1 + oXi1))
  }
  return(L1)
}
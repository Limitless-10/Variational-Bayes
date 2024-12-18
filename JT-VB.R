# Entropy for multivariate normal
mvn_ent <- function(Sigma) {
  (ncol(Sigma)*(1 + log(2*pi)) + log(det(Sigma)))/2
}

# Entropy for inverse gamma
ig_ent <- function(a, b) {
  a + log(b) + lgamma(a) - (a + 1)*digamma(a)
}

# VB for linear regression model
vb_lin_reg <- function(
    X, y, 
    mu0 = rep(0, ncol(X)), Sigma0 = diag(10, ncol(X)), 
    a0 = 1e-2, b0 = 1e-2, 
    maxiter = 100, tol = 1e-5, verbose = TRUE) {
  
  d <- ncol(X)
  n <- nrow(X)
  invSigma0 <- solve(Sigma0)
  invSigma0_x_mu0 <- invSigma0 %*% mu0
  XtX <- crossprod(X)
  Xty <- crossprod(X, y)
  mu <- mu0
  Sigma <- Sigma0
  a <- a0 + n / 2
  b <- b0
  lb <- as.numeric(maxiter)
  i <- 0
  converged <- FALSE
  while(i <= maxiter & !converged) {
    i <- i + 1
    a_div_b <- a / b
    
    Sigma <- solve(a_div_b * XtX + invSigma0)
    mu <- Sigma %*% (a_div_b * Xty + invSigma0_x_mu0)
    y_m_Xmu <- y - X %*% mu
    b <- b0 + 0.5*(crossprod(y_m_Xmu) + sum(diag(Sigma %*% XtX)))[1]
    
    # Calculate L(q)
    lb[i] <- mvn_ent(Sigma) + ig_ent(a, b) +
      a0 * log(b0) - lgamma(a0) - (a0 + 1) * (log(b) - digamma(a)) - b0 * a / b -
      0.5*(d * log(2*pi) + log(det(Sigma0)) + crossprod(mu - mu0, Sigma0 %*% (mu - mu0)) + sum(diag(invSigma0 %*% Sigma))) -
      0.5*(n * log(2*pi) - log(b) + digamma(a) + a / b * (crossprod(y_m_Xmu) + sum(diag(XtX %*% Sigma))))
    
    if(verbose) cat(sprintf("Iteration %3d, ELBO = %5.10f\n", i, lb[i]))
    if(i > 1 && abs(lb[i] - lb[i - 1]) < tol) converged <- TRUE
  }
  return(list(lb = lb[1:i], mu = mu, Sigma = Sigma, a = a, b = b))
}
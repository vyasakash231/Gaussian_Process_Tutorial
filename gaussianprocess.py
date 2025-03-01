class GaussianProcess:
    """
    Gaussian processes is a way to define distributions over functions of the form f:X → R, where X is of size (NxD). 
    The key assumption is that the function values at a set of M > 0 inputs, F = [f(x1),...,f(xM)], is jointly Gaussian, with mean (µ = m(x1),...,m(xM))
    and covariance Σij = K(xi, xj), where "m" is a mean function and K is a positive definite (Mercer) kernel. 
    Since we assume this holds for any M > 0, this includes the case where M = N + h, containing N training points (X,Y) where {X=[x1,x2,..,xN], Fx=[y1,y2,..,yN]} 
    and h test point X*=[x*1,x*2,..,x*N], where X* is of size (hxD).
    where, Y = Fx. Thus we can infer f(X*) from knowledge of f(x1),...,f(xN) by manipulating the joint Gaussian distribution p(f(x1),...,f(xN),f(X*)) like this.
    
    So, by defination, the joint distribution p(Fx,F*|X,X*) has the form:
     /Fx\      //µx\    /K(X,X)     K(X,X*)\\
    |    | ~ N||    |, |                    ||, so K(X,X) is NxN, K(X,X*) is Nxh, K(X*,X*) is hxh matrices.
     \F*/      \\µ*/    \K(X*,X)   K(X*,X*)//
    where, µx = [m(x1),...,m(xM)], µ* = [m(x*1),...,m(x*M)], K(X*,X) = K(X,X*).T 

    We can also extend this to work with the case where we observe noisy functions of f(xn), such as in regression or classification problems.

    Now, by the standard rules of conditioning gaussias, the posterior has the following form: p(F*|X*,D) = p(F*|X*,X,Fx)=> N(F*|µ*,Σ*)
    µ* = m(X*) + K(X,X*).T * K(X,X)^(-1) * (Fx - m(X))
    Σ* = K(X*,X*) - K(X,X*)^T * K(X,X)^(-1) * K(X,X*)

    where, m(X) ->  is the prior mean function evaluated at the training points X.
    m(X*) -> the prior mean function evaluated at the test points X*. It's our initial guess of the mean value at X* before seeing any training data.
    """
    def __init__(self, X=None):
        self.X = X
        self.X_train = None
        self.Y_train = None
        self.mu_prior_train = None

    def set_training_data(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.mu_prior_train = np.zeros(self.Y_train.shape)

    def gaussian_kernel(self, xi, xj, l=1.0, sigma=1.0):  
        """
        It tell's us, how the points xi, xj are correlated, point which are very close are higly correlated and points which are far from each other 
        are less correlated, and this correlation will have a 2D gaussian type bell curve, so, points clossest will have highest kernel value and points 
        far away will have small kernel value and this is based on sigma**2 * np.exp(-(0.5 * euclid_dist) / l**2) formula of gaussian PDF, 
        l is similar to std_dev, so let's say if l=0.5 then points which have small kernel value (small correlation) will increase, if we increase l 
        to let's say 2.0 (it's like gaussian surface is bulging/expanding)
        """
        xi = np.atleast_2d(xi)  # (N, D) where N is the no of of points
        xj = np.atleast_2d(xj)  # (M, D) where M is the no of of points
        
        # euclid_dist = np.zeros((xi.shape[0], xj.shape[0])) # Initialize an empty matrix to store distances

        # for i in range(xi.shape[0]):       # Loop through each point in x1
        #     for j in range(xj.shape[0]):   # Loop through each point in x2
        #         squared_distance = np.sum((xi[i] - xj[j]) ** 2) # Sum of squared components (squared Euclidean distance)
        #         euclid_dist[i, j] = squared_distance # Store the distance in the matrix

        euclid_dist = np.sum((xi[:, np.newaxis, :] - xj[np.newaxis, :, :]) ** 2, axis=2)

        kernel = sigma**2 * np.exp(-(0.5 * euclid_dist) / l**2)   # sigma**2 -> variance, l -> lengthscale
        return kernel
    
    def prior(self):
        """
        Another way to find prior is to use cholesky decomposition, which is a way to decompose a matrix into a product of a lower triangular matrix and its transpose.
        Σ = LL^T, where L is a lower triangular matrix. We can then sample from a multivariate gaussian with mean µ and covariance Σ by sampling from a standard normal distribution,
        and then multiplying by L and adding µ.

        /f1\      //µ1\    /Σ11   Σ12\\              /µ1\           /Σ11   Σ12\
       |    | ~ N||    |, |           ||, where µ = |    | and Σ = |           |
        \f2/      \\µ2/    \Σ21   Σ22//              \µ2/           \Σ21   Σ22/

        f ~ N(µ,Σ) => f = µ + L*z, where z ~ N(0,1)
        """
        mu_prior = np.zeros(self.X.shape[0])
        covariance_prior = self.gaussian_kernel(self.X, self.X)
        return mu_prior, covariance_prior

    def posterior_using_kernel_inv(self, X_test, sigma_n=0.1):
        """
        here, 1 -> training data, 2 -> test data
        The marginal distribution are given by:
        p(f1) = N(f1|µ1,Σ11)
        p(f2) = N(f2|µ2,Σ22)

        and posterior distribution is given by:
        p(f2|f1) = N(f2|µ_{2|1}, Σ_{2|1})
        """
        self.mu_prior_test = np.zeros(X_test.shape[0])

        K11 = self.gaussian_kernel(self.X_train, self.X_train)
        K12 = self.gaussian_kernel(self.X_train, X_test)
        K22 = self.gaussian_kernel(X_test, X_test)

        """For Noisy GP regression, where, yn = f(xn) + ϵn, where ϵn ~ N(0,σ^2) is a gaussian noise."""
        Ky = K11 + sigma_n**2 * np.eye(len(self.X_train))  # Ky = K11 + σ^{2} * I, where I is the identity matrix
        K11_inv = np.linalg.inv(Ky) 

        # µ_{2|1} = µ2 + Σ21 * Σ11^{-1} * (f1 - µ1)
        self.mu_posterior_test = self.mu_prior_test + K12.T @ K11_inv @ (self.Y_train - self.mu_prior_train)

        # Σ_{2|1} = Σ22 - Σ21 * Σ11^{-1} * Σ12
        self.covariance_posterior = K22 - K12.T @ K11_inv @ K12

        return self.mu_posterior_test, self.covariance_posterior

    def posterior_using_cholesky(self, X_test, sigma_n=0.1):
        self.mu_prior_test = np.zeros(X_test.shape[0])

        K11 = self.gaussian_kernel(self.X_train, self.X_train)
        K12 = self.gaussian_kernel(self.X_train, X_test)
        K22 = self.gaussian_kernel(X_test, X_test)

        """
        For Noisy GP regression, where, yn = f(xn) + ϵn, where ϵn ~ N(0,σ^2) is a gaussian noise
        For reasons of numerical stability, it is unwise to directly invert K11. A more robust alternative 
        is to compute a Cholesky decomposition, Σ11 = K11 = L*L^T , which takes O(N^3) time.
        µ_{2|1} = µ2 + Σ21 * Σ11^{-1} * (f1 - µ1) = µ2 + Σ21 * α
        Σ_{2|1} = Σ22 - Σ21 * Σ11^{-1} * Σ12 = Σ22 - v^T * v
        
        where, α = Σ11^{-1} * (f1 - µ1) and v = L^{-1} * Σ12
        α = Σ11^{-1} * (f1 - µ1) = L^{-T} * L^{-1} * (f1 - µ1) = L^{-T} * m
        where, m = L^{-1} * (f1 - µ1) => L*m = (f1 - µ1) 

        so, first find m by solving L*m = (f1 - µ1), then find α by solving L^{T}*α = m, this will give us µ_{2|1} 
        now, to find Σ_{2|1}, first solve, L*v = Σ12, then find Σ_{2|1} = Σ22 - v^T * v
        """
        L = np.linalg.cholesky(K11 + sigma_n**2 * np.eye(len(self.X_train))) 
        m = np.linalg.solve(L, self.Y_train - self.mu_prior_train)
        alpha = np.linalg.solve(L.T, m)
        v = np.linalg.solve(L, K12)
        
        self.mu_posterior_test = self.mu_prior_test + K12.T @ alpha
        self.covariance_posterior = K22 - v.T @ v
        return self.mu_posterior_test, self.covariance_posterior
    
    def sample_multivariate_gaussian(self, mu, cov, n_samples):
        return np.random.multivariate_normal(mu, cov, size=n_samples)  # drawn samples, of shape "size", if it was provided
    
    def predict(self, X_test, return_std=True, n_std=1.0):
        mu, cov = self.posterior_using_kernel_inv(X_test)
        # mu, cov = self.posterior_using_cholesky(X_test)
        if return_std:
            std = np.sqrt(np.diag(cov))
            return mu, mu - n_std * std, mu + n_std * std
        return mu

    def plot_prior(self, n_sample, idx):
        plt.subplot(1, 2, idx)
        plt.title('GP Prior Samples (N=0)')
        
        # --- GP Prior Distribution ---
        mu_prior, cov_prior = self.prior()
        std_prior = np.sqrt(np.diag(cov_prior))
        
        prior_samples = self.sample_multivariate_gaussian(mu_prior, cov_prior, n_sample)
        
        for i in range(n_sample):
            plt.plot(self.X, prior_samples[i], color="green", lw=1)
        plt.fill_between(self.X.flatten(), mu_prior - 2*std_prior, mu_prior + 2*std_prior, color='gray', alpha=0.3, label='±1 std')
        plt.plot(self.X, mu_prior, color="black", lw=2)
        plt.ylabel("N=0")
        plt.xlim([-5, 5])
        plt.ylim([-3, 3])

    def plot_posterior(self, n_sample, idx):
        # --- GP Posterior Distribution ---
        plt.subplot(1, 2, idx)
        plt.title(f'GP Posterior (N={self.X_train.shape[0]})')
        
        mu_posterior, cov_posterior = self.posterior_using_kernel_inv(self.X)
        # mu_posterior, cov_posterior = self.posterior_using_cholesky(self.X)
        std_posterior = np.sqrt(np.diag(cov_posterior))
        
        posterior_samples = self.sample_multivariate_gaussian(mu_posterior, cov_posterior, n_sample)
        
        for i in range(n_sample):
            plt.plot(self.X, posterior_samples[i], color="green", lw=1)  # plot samples from posteriors
        plt.fill_between(self.X.flatten(), mu_posterior - 2*std_posterior, mu_posterior + 2*std_posterior, color='gray', alpha=0.3, label='±1 std')
        plt.scatter(self.X_train, self.Y_train, color='red', s=50, marker='o', label='Training Data')  # plot training data points
        plt.plot(self.X, mu_posterior, color="black", lw=2)
        plt.ylabel(f"N={self.X_train.shape[0]}")
        plt.xlim([-5, 5])
        plt.ylim([-3, 3])

    def plot(self, n_sample):
        plt.figure(figsize=(10, 8))
        self.plot_prior(n_sample, 1)
        self.plot_posterior(n_sample, 2)
        plt.show()


if __name__ == "__main__":
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    GP = GaussianProcess(X)

    """Train GP Model"""
    # X_train = np.array([-3.0]).reshape(-1, 1)
    # Y_train = np.array([1.5])

    X_train = np.array([-3.0, 2.0]).reshape(-1, 1)
    Y_train = np.array([1.5, -0.5])

    # X_train = np.array([-3.0, -1.0, 1.0, 3.0]).reshape(-1, 1)
    # Y_train = np.array([1.5, -0.5, -1.0, 0.5])

    GP.set_training_data(X_train, Y_train)

    """Test GP Model"""
    X_test = np.array([-2.0]).reshape(-1, 1)
    Y_predict = GP.predict(X_test, return_std=False)
    print(Y_predict)

    GP.plot(n_sample=3)

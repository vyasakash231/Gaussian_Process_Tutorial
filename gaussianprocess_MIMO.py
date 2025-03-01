import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv

class MIMOGaussianProcess:
    """
    Multi-Input Multi-Output Gaussian Process implementation using the Linear Model of Coregionalization (LMC) approach.
    
    This implementation extends the single-output GP to handle multiple outputs by modeling the correlations between
    different output dimensions. The key idea is to construct a valid covariance function that captures both:
    1. The spatial correlation between input points (as in single-output GP)
    2. The correlation between different outputs
    
    The covariance function takes the form:
    k((x,i), (x',j)) = B[i,j] * k_spatial(x, x')
    
    where:
    - x, x' are input vectors
    - i, j are output indices
    - B is the coregionalization matrix (captures output correlations)
    - k_spatial is the base kernel function for spatial correlation
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = None
        self.Y_train = None
        self.B = np.eye(output_dim)  # Initialize coregionalization matrix

        # Default hyperparameters
        self.hyperparams = {
            'l': 2.0,           # Length scale
            'sigma_f': 1.0,     # Signal variance
            'sigma_n': 1e-6    # Noise standard deviation
        }
        
    def set_training_data(self, X_train, Y_train):
        """
            X_train: Training inputs of shape (N, input_dim)
            Y_train: Training outputs of shape (N, output_dim)
        """
        assert X_train.shape[1] == self.input_dim, f"Expected {self.input_dim} input dimensions, got {X_train.shape[1]}"
        assert Y_train.shape[1] == self.output_dim, f"Expected {self.output_dim} output dimensions, got {Y_train.shape[1]}"
        
        self.X_train = X_train
        self.Y_train = Y_train
        
    def gaussian_kernel(self, xi, xj, l=1.0, sigma=1.0):
        xi = np.atleast_2d(xi)
        xj = np.atleast_2d(xj)
        
        # Compute squared Euclidean distances efficiently
        euclid_dist = np.sum((xi[:, np.newaxis, :] - xj[np.newaxis, :, :]) ** 2, axis=2)
        return sigma**2 * np.exp(-(0.5 * euclid_dist) / l**2)
    
    '''https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/gaussian_process/kernels.py#L1598'''
    def matern_kernel(self, xi, xj, nu=1.5, l=1.0, sigma=1.0):
        """
        Compute the Matern kernel between two sets of points.
        
        Parameters:
        - xi: Array of shape (n, d), where n is the number of points and d is the dimensionality.
        - xj: Array of shape (m, d), where m is the number of points and d is the dimensionality.
        
        Returns:
        - Kernel matrix of shape (n, m).
        """
        xi = np.atleast_2d(xi)
        xj = np.atleast_2d(xj)
        
        # Compute pairwise Euclidean distances
        dist = np.sqrt(np.sum((xi[:, np.newaxis, :] - xj[np.newaxis, :, :]) ** 2, axis=2))
        
        # Avoid division by zero for zero distances
        dist[dist == 0] = 1e-8
        
        # Precompute constants
        sqrt_2nu = np.sqrt(2 * nu)
        scaled_dist = sqrt_2nu * dist / l
        
        # Compute the Matern kernel
        if nu == 0.5:
            # Matern 1/2 kernel (equivalent to exponential kernel)
            K = sigma**2 * np.exp(-scaled_dist)
        elif nu == 1.5:
            # Matern 3/2 kernel
            K = sigma**2 * (1 + scaled_dist) * np.exp(-scaled_dist)
        elif nu == 2.5:
            # Matern 5/2 kernel
            K = sigma**2 * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
        else:
            # General Matern kernel for any nu > 0
            term1 = (2 ** (1 - self.nu)) / gamma(self.nu)
            term2 = (sqrt_2nu * dist / self.l) *
    
    def mimo_kernel(self, X1, X2, l=None, sigma_f=None):
        """
        Compute the full MIMO kernel matrix incorporating both spatial and output correlations.
        
        Args:
            X1: First set of inputs (N, input_dim)
            X2: Second set of inputs (M, input_dim)
            
        Returns:
            Full kernel matrix of shape (N*output_dim, M*output_dim)
        """
        if l is None:
            l = self.hyperparams['l']
        if sigma_f is None:
            sigma_f = self.hyperparams['sigma_f']

        # K_spatial = self.gaussian_kernel(X1, X2, l, sigma_f)
        K_spatial = self.matern_kernel(X1, X2, nu=2.5, l=l, sigma=sigma_f)
        
        # Construct the full MIMO kernel using Kronecker product
        K_full = np.kron(K_spatial, self.B)
        return K_full
    
    def posterior(self, X_test, sigma_n=None):
        """
        Compute the posterior distribution for test points.
        """
        if sigma_n is None:
            sigma_n = self.hyperparams['sigma_n']

        N_train = self.X_train.shape[0]
        N_test = X_test.shape[0]
        
        # Compute kernel matrices
        K_train = self.mimo_kernel(self.X_train, self.X_train)
        K_test = self.mimo_kernel(X_test, X_test)
        K_cross = self.mimo_kernel(self.X_train, X_test)
        
        # Add noise to training kernel
        K_train = K_train + sigma_n**2 * np.eye(N_train * self.output_dim)  # Ky = K11 + σ^{2} * I, where I is the identity matrix
        
        # Reshape Y_train for MIMO calculations
        Y_train_flat = self.Y_train.reshape(-1)
        
        try:
            # Compute posterior using Cholesky decomposition for stability
            L = np.linalg.cholesky(K_train)
            m = np.linalg.solve(L, Y_train_flat)
            alpha = np.linalg.solve(L.T, m)
            
            # Compute posterior mean and covariance
            mu_post = K_cross.T @ alpha
            v = np.linalg.solve(L, K_cross)
            cov_post = K_test - v.T @ v
            
            # Add a small jitter to ensure positive definiteness
            cov_post = cov_post + 1e-8 * np.eye(N_test * self.output_dim)
            
            # Reshape posterior mean back to (M, output_dim)
            mu_post = mu_post.reshape(N_test, self.output_dim)
            return mu_post, cov_post
        
        except np.linalg.LinAlgError:
            # Fall back to more stable but slower method if Cholesky fails
            print("Warning: Cholesky decomposition failed in posterior calculation.")
            
            # Use pseudoinverse instead
            K_inv = np.linalg.pinv(K_train)
            mu_post = (K_cross.T @ K_inv @ Y_train_flat).reshape(N_test, self.output_dim)
            cov_post = K_test - K_cross.T @ K_inv @ K_cross + 1e-8 * np.eye(N_test * self.output_dim)
            return mu_post, cov_post
    
    def predict(self, X_test, return_std=True, n_std=2.0):
        mu, cov = self.posterior(X_test, sigma_n=0.01)
        if return_std:
            std = np.sqrt(np.diag(cov)).reshape(-1, self.output_dim)
            return mu, mu - n_std * std, mu + n_std * std
        return mu
    
    def fit_coregionalization_matrix(self, method: str = 'empirical'):
        """
        From "Kernels for Vector-Valued Functions: a Review"
        Fit the coregionalization matrix B using training data.
        method: Method to use for fitting ('empirical' or 'mle')     
        """
        if method == 'empirical':
            # Simple empirical estimation using output correlations
            self.B = np.corrcoef(self.Y_train.T)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        
    def log_maximum_likelihood(self, params):  # params: Hyperparameters [log(l), log(sigma_f), log(sigma_n)]
        """
        Compute the log marginal likelihood of the training data.
        log(p(y|X,θ)) = - 0.5 * y^T * Ky^{-1} * y - 0.5 * log|K11| - N/2 * log(2*pi)  
        Ky = K11 + σ^{2} * I, where I is the identity matrix
        """
        # Extract and transform parameters (work with log values for optimization stability)
        l = np.exp(params[0])
        sigma_f = np.exp(params[1])
        sigma_n = np.exp(params[2])

        # Ensure minimum noise to prevent singular matrices
        sigma_n = max(sigma_n, 1e-6)

        # Reshape Y_train for MIMO calculations
        Y_train_flat = self.Y_train.reshape(-1)

        try:
            # Compute kernel matrix
            K = self.mimo_kernel(self.X_train, self.X_train, l, sigma_f)
            n = K.shape[0]
            
            # Use Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # If Cholesky fails, add jitter and try again
                K = K + 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
            
            # Compute log determinant using Cholesky (more stable)
            log_det_K = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve linear system using Cholesky
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_flat))
            
            # Compute log likelihood
            log_likelihood = - 0.5 * Y_train_flat.T @ alpha - 0.5 * log_det_K - 0.5 * n * np.log(2 * np.pi)
            
            # Return negative log likelihood (for minimization)
            return -log_likelihood
        
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
            # Handle any other numerical errors
            print(f"Warning in log likelihood calculation: {e}")
            return 1e10

    def plot_2d_outputs(self, X_test, output_names=None):
        if self.output_dim != 2:
            raise ValueError("This plotting function is only for 2D outputs")
            
        mu, lower, upper = self.predict(X_test)
        
        if output_names is None:
            output_names = [f'Output {i+1}' for i in range(2)]
            
        plt.figure(figsize=(12, 5))
        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.plot(X_test[:, 0], mu[:, i], 'g-', label='Mean prediction')
            plt.fill_between(X_test[:, 0], lower[:, i], upper[:, i], color='grey', alpha=0.5, label='95% confidence')
            if self.X_train is not None:
                plt.scatter(self.X_train[:, 0], self.Y_train[:, i], c='r', marker='o', label='Training data')
            plt.xlabel('Input')
            plt.ylabel(output_names[i])
            plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Generate some synthetic data for a 2D input, 2D output problem
    X_train = np.random.rand(50, 2) * 10  # 50 points, 2D input
    
    # Generate correlated outputs
    Y1 = np.sin(0.5 * X_train[:, 0]) + 0.1 * np.random.randn(50)
    Y2 = np.cos(0.5 * X_train[:, 0] + 0.5) + 0.2 * np.sin(X_train[:, 1]) + 0.05 * np.random.randn(50)
    Y_train = np.column_stack([Y1, Y2])
    
    # Create and train the MIMO GP
    gp = MIMOGaussianProcess(input_dim=2, output_dim=2)
    gp.set_training_data(X_train, Y_train)
    
    # Fit the coregionalization matrix
    gp.fit_coregionalization_matrix()

    # Generate test points
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    X_test = np.hstack([X_test, 5 * np.ones_like(X_test)])  # Fix second dimension at 5
    
    # Make predictions and plot
    gp.plot_2d_outputs(X_test, output_names=['sin function', 'cos function'])

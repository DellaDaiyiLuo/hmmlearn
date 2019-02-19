import numpy as np
from scipy import linalg
from scipy.special import gammaln


def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.
    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.
    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.
    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'
    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.
    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars
    )


def _log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model."""
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr


def _log_multivariate_normal_density_spherical(X, means, covars):
    """Compute Gaussian log-density at X for a spherical model."""
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:, np.newaxis]
    if cv.shape[1] == 1:
        cv = np.tile(cv, (1, X.shape[-1]))
    return _log_multivariate_normal_density_diag(X, means, cv)


def _log_multivariate_normal_density_tied(X, means, covars):
    """Compute Gaussian log-density at X for a tied model."""
    cv = np.tile(covars, (means.shape[0], 1, 1))
    return _log_multivariate_normal_density_full(X, means, cv)


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob

def log_multivariate_poisson_density(X, means) :
  # # modeled on log_multivariate_normal_density from sklearn.mixture
    #print("X has shape {}".format(X.shape))
    #print("means has shape {}".format(means.shape))
    n_samples, n_dim = X.shape
    # -lambda + k log(lambda) - log(k!)
    log_means = np.where(means > 1e-3, np.log(means), np.log(1e-3))
    lpr =  np.dot(X, log_means.T)
    lpr = lpr - np.sum(means,axis=1) # rates for all elements are summed and then broadcast across the observation dimenension
    log_factorial = np.sum(gammaln(X + 1), axis=1)
    lpr = lpr - log_factorial[:,None] # logfactobs vector broadcast across the state dimension

    #print("lpr has shape {}".format(lpr.shape))
    return lpr

def log_multivariate_gamma_density(X, shape, scale):

    """"

    Parameters
    ==========
    X : np.ndarray, shape (n_samples, n_features)
        Input data
    shape : np.ndarray, shape (n_components, n_features)
        The shape parameters of the Gamma distribution,
        also known as k
    scale : np.ndarray, shape (n_components, n_features)
        The scale parameter of the Gamma distribution,
        also known as theta

    Returns
    =======
    lpr : ndarray, shape (n_samples, n_components)
        Log likelihood of the data, for each component
        (latent state) given the parameters
    """

    # The log likelihood is given by 
    # log L(alpha, beta|X) = 
    #     alpha*log(beta) - log(gamma(alpha)) + (alpha - 1)*log(X) - beta*X
    # For X with shape (1, n_features), and parameters with 
    # shape (n_components, n_features), this results in an output of shape
    # (n_components, n_features). Assuming each feature is independent of
    # the others, we sum the output array across all features (equivalent to
    # multiplying probabilities in the non-log domain) to obtain an output
    # of shape (n_components, ). We then reshape it to be (1, n_components).
    # If X contains multiple samples, this strategy generalizes; the output
    # will then have shape (n_samples, n_components)

    alpha = shape
    beta = 1/scale
    alpha = np.where(alpha > 1e-3, alpha, 1e-3)
    beta = np.where(beta > 1e-3, beta, 1e-3)

    assert alpha.shape == beta.shape, "Alpha and beta have mismatched dimensions"

    # Break up log likelihood calculation into terms
    term1 = (alpha*np.log(beta)).sum(axis=1, keepdims=True).T
    term2 = gammaln(alpha).sum(axis=1, keepdims=True).T
    term3 = np.dot(np.log(X), (alpha - 1).T)
    term4 = np.dot(X, beta.T)

    lpr = term1 - term2 + term3 - term4
    return lpr

def log_marked_poisson_density(X, rates, n_clusters):
    """Compute the log probability under a multivariate Gaussian marked
    Poisson 'distribution'.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_clusters)
        List of n_clusters-dimensional data points. Each row corresponds to a
        single data point.
    rates : array_like, shape (n_components, n_clusters)
        List of n_clusters-dimensional relative rate vectors for n_components.
        Each row corresponds to a single rate vector.
    n_clusters : int
        Dimensionality of the latent (independent) Poisson emissions.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each sample in
        X under each of the n_components mark distributions.
    """

    raise NotImplementedError
    # multiprocessing here for sampling-based evaluation

    return lpr
import numpy as np
from scipy import linalg
from scipy.special import logsumexp, gammaln
from scipy.stats import multivariate_normal, poisson

MIN_LIKELIHOOD = 1e-300
MIN_LOGLIKELIHOOD = -700

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

def sample_IKR(r, *, K=None, N=None, M=None, mode='id'):
    """Sample I^K|R.

    Note that 1^K|R is called I^K|R because we cannot have variables
    that start with a number in Python.

    Parameters
    ----------
    r : array-like
        Relative rates of shape (N,) for current time window, and state
    K : int, optional
        Number of events to sample, default is one.
    N : int, optional
        Number of neurons, if not specified, will be obtained from R.
    M : int, optional
        Number of length-K samples to generate. Default is one.
    mode : str, optional
        String specifying output mode. Default is 'id'.
        Alternative is 'ek', which returns the sequence of vectors.

    Returns
    -------
    ikr : samples with shape (M, K) if mode=='id', and with shape (M, N, K)
        if mode=='ek'

    Example
    -------
    >>> r = [0.1, 0.4, 0.2]
    >>> ikr = sample_IKR(r=r, K=30, M=20, mode='id')

    """

    if K is None:
        K = 1
    if M is None:
        M = 1
    if N is None:
        N = len(r)

    p = r / np.sum(r) # only want to do this once!

    if mode == 'id':
        ikr = np.random.choice(a=N, size=K*M, p=p )
        ikr = np.reshape(ikr, (M, K))
    elif mode == 'ek':
        ikr = np.random.multinomial(n=1, pvals=p, size=K*M)
        ikr = np.reshape(ikr, (K, M, N)).transpose([1,2,0])
    else:
        raise ValueError("mode '{}' not understood.".format(mode))

    return ikr

def eval_mark_loglikelihoods(*, marks, ikr, mu, Sigma, rates):
    """
    Compute P(Y=marks | I^K), where I^K ~ rates.

    Strategy: first pre-compute the NxK likelihood of observing
    each mark from every neuron. Then use this matrix to compute
    the rest.

    Parameters
    ----------
    marks : array-like, with shape (K, D)
        Observed marks, with shape (K, D), where D is the dimensionality
        of the mark space, and K is the number of observed marks.

    ikr : array-like, with shape (M, K)
        Sampled neuron IDs for each sample, and each mark.
    mu : array-like, with shape (N, D)
        D-dimensional means for each of the N neurons.
    Sigma : array-like, with shape (N, D, D)
        D-by-D-dimensional covariances for each of the N neurons.
    rates : array-like, with shape (N,)
        Rates for each of the neurons.

    Returns
    -------
    ll : log likelihoods for each sample. Shape (M,)
    """
    N = len(mu)
    M, K = ikr.shape

    logF = np.zeros((N, K)) # precomputed multivariate normal probs
    logP = np.zeros((N, K+1)) # precomputed Poisson probs

    n_marks = np.arange(K+1)
    for nn in range(N):
        # multivariate normal
        mvn = multivariate_normal(mean=mu[nn], cov=Sigma[nn])
        f = np.atleast_1d(mvn.logpdf(marks))
        f[f < MIN_LOGLIKELIHOOD] = MIN_LOGLIKELIHOOD
        logF[nn,:] = f
        # poisson
        pois = poisson(rates[nn])
        p = np.atleast_1d(pois.logpmf(n_marks))
        p[p < MIN_LOGLIKELIHOOD] = MIN_LOGLIKELIHOOD
        logP[nn,:] = p

    bins = np.arange(N+1)
    ll = np.zeros(M)
    krange = np.arange(K)
    nrange = np.arange(N)
    for ii in range(M):
        Vii = np.histogram(ikr[ii], bins=bins)[0]
        ll[ii] = np.sum(logF[ikr[ii], krange]) + np.sum( logP[nrange, Vii] )

    return ll

def eval_P_Y_given_ISR(*, marks, rates, mu, Sigma, M):
    """
    Appriximate P(Y=marks | r_t^{(j)}) by sampling I^K|R.

    We use logsumexp to make this numerically more stable.

    Parameters
    ----------
    marks : array-like, with shape (K, D)
        Observed marks, with shape (K, D), where D is the dimensionality
        of the mark space, and K is the number of observed marks.
    rates : array-like
        Relative rates of shape (N,) for current time window, and state
    mu : array-like, with shape (N, D)
        D-dimensional means for each of the N neurons.
    Sigma : array-like, with shape (N, D, D)
        D-by-D-dimensional covariances for each of the N neurons.

    Returns
    -------
    logP : log probability of observing sequence of marks

    """
    K = len(marks)

    ikr = sample_IKR(r=rates,
                    K=K,
                    M=M,
                    mode='id')

    ll = eval_mark_loglikelihoods(marks=marks,
                                ikr=ikr,
                                mu=mu,
                                Sigma=Sigma,
                                rates=rates)

    logP = logsumexp(ll) - np.log(M)

    return logP

def log_marked_poisson_density(X, rates, cluster_means, cluster_covars, n_samples):
    """Compute the log probability under a multivariate Gaussian marked
    Poisson 'distribution'.

    Parameters
    ----------
    X : array_like, shape (n_obs, )
        List of mark features. Each element is a (K, D) array, where K is the
        number of marks in the particular observation, and D is the mark feature
        dimension.
    rates : array_like, shape (n_components, n_clusters)
        List of n_clusters-dimensional relative rate vectors for n_components.
        Each row corresponds to a single rate vector.

    Returns
    -------
    lpr : array_like, shape (n_obs, n_components)
        Array containing the log probabilities of each sample in
        X under each of the n_components mark distributions.
    """

    import multiprocessing as mp
    import psutil

    n_processes = psutil.cpu_count()
    # pool = mp.Pool(processes=n_processes)

    n_components, n_clusters = rates.shape
    n_obs = len(X)
    lpr = np.zeros((n_obs, n_components))

    for zz in range(n_components):
        r = rates[zz,:].squeeze()
        with mp.Pool(processes=n_processes) as pool:
            results = [pool.apply_async(eval_P_Y_given_ISR,
                kwds={'marks' : obs,
                    'rates' : r,
                    'mu': cluster_means,
                    'Sigma' : cluster_covars,
                    'M' : n_samples}) for obs in X]
            results = [p.get() for p in results]
        lpr[:,zz] = results

    lprs = np.sum(lpr, axis=1)
    lpr[lprs==0,:] = np.log(1/n_components)

    return lpr
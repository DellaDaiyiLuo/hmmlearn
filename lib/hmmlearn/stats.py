import numpy as np
from scipy import linalg
from scipy.special import logsumexp, gammaln
from scipy.stats import multivariate_normal, poisson
from sklearn.utils import check_random_state

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

def sample_IKR(rates, *, n_marks=None, n_clusters=None, n_samples=None, mode='id', random_state=None):
    """Sample I^K|R.

    Note that 1^K|R is called I^K|R because we cannot have variables
    that start with a number in Python.

    Parameters
    ----------
    rates : array-like
        Relative rates of shape (n_clusters,) for current time window, and state
    n_marks : int, optional
        Number of events to sample, default is one.
    n_clusters : int, optional
        Number of neurons, if not specified, will be obtained from R.
    n_samples : int, optional
        Number of length-n_marks samples to generate. Default is one.
    mode : str, optional
        String specifying output mode. Default is 'id'.
        Alternative is 'ek', which returns the sequence of vectors.

    Returns
    -------
    ikr : samples with shape (n_samples, n_marks) if mode=='id', and with
        shape (n_samples, n_clusters, n_marks) if mode=='ek'

    Example
    -------
    >>> r = [0.1, 0.4, 0.2]
    >>> ikr = sample_IKR(rates=r, n_marks=30, n_samples=20, mode='id')

    """
    if n_marks is None:
        n_marks = 1
    if n_samples is None:
        n_samples = 1
    if n_clusters is None:
        n_clusters = len(rates)

    p = rates / np.sum(rates)

    rng = check_random_state(random_state)
    if mode == 'id':
        ikr = rng.choice(a=n_clusters, size=n_marks*n_samples, p=p )
        ikr = np.reshape(ikr, (n_samples, n_marks))
    elif mode == 'ek':
        ikr = rng.multinomial(n=1, pvals=p, size=n_marks*n_samples)
        ikr = np.reshape(ikr, (n_marks, n_samples, n_clusters)).transpose([1,2,0])
    else:
        raise ValueError("mode '{}' not understood.".format(mode))

    return ikr

def eval_mark_loglikelihoods(*, marks, ikr, cluster_means, cluster_covars, rates, reorder=False):
    """
    Compute P(Y=marks | I^K), where I^K ~ rates.

    Strategy: first pre-compute the n_clusters x n_marks likelihood of observing
    each mark from every neuron. Then use this matrix to compute
    the rest.

    Parameters
    ----------
    marks : array-like, with shape (n_marks, D)
        Observed marks, with shape (n_marks, D), where D is the dimensionality
        of the mark space, and n_marks is the number of observed marks.

    ikr : array-like, with shape (n_samples, n_marks)
        Sampled neuron IDs for each sample, and each mark.
    cluster_means : array-like, with shape (n_clusters, D)
        D-dimensional means for each of the n_clusters neurons.
    cluster_covars : array-like, with shape (n_clusters, D, D)
        D-by-D-dimensional covariances for each of the n_clusters neurons.
    rates : array-like, with shape (n_clusters,)
        Rates for each of the neurons.
    reorder : boolean, optional (default is False)
        If True, then the samples IKR will be reordered to maximize the
        likelihood. The ordering is not guaranteed to be the glabally optimal
        ordering (at least not at this time), but it is guaranteed to be some
        locally (greedy) optimal solution.

    Returns
    -------
    ll : log likelihoods for each sample. Shape (n_samples,)
    """
    n_clusters = len(cluster_means)
    n_samples, n_marks = ikr.shape

    logF = np.zeros((n_clusters, n_marks)) # precomputed multivariate normal probs
    logP = np.zeros((n_clusters, n_marks+1)) # precomputed Poisson probs

    mark_range = np.arange(n_marks+1)
    for nn in range(n_clusters):
        # multivariate normal
        mvn = multivariate_normal(mean=cluster_means[nn], cov=cluster_covars[nn])
        f = np.atleast_1d(mvn.logpdf(marks))
        f[f < MIN_LOGLIKELIHOOD] = MIN_LOGLIKELIHOOD
        logF[nn,:] = f
        # poisson
        pois = poisson(rates[nn])
        p = np.atleast_1d(pois.logpmf(mark_range))
        p[p < MIN_LOGLIKELIHOOD] = MIN_LOGLIKELIHOOD
        logP[nn,:] = p

    bins = np.arange(n_clusters+1)
    ll = np.zeros(n_samples)
    krange = np.arange(n_marks)
    nrange = np.arange(n_clusters)
    for ii in range(n_samples):
        Vii = np.histogram(ikr[ii], bins=bins)[0]
        if reorder: # TODO: make this more performant!!! (tricky?)
            neworder = []
            remaining_midx = list(range(n_marks))
            for mm in range(n_marks):
                 midx = remaining_midx[np.argmax(logF[ikr[ii,mm], remaining_midx])]
                 neworder.append(midx)
                 remaining_midx.remove(midx)
            ikr[ii] = ikr[ii][neworder]
            pass
        ll[ii] = np.sum(logF[ikr[ii], krange]) + np.sum( logP[nrange, Vii] )

    return ll

def eval_P_Y_given_ISR(*, marks, rates, cluster_means, cluster_covars,
                        n_samples=None, stype='unbiased', reorder=False,
                        random_state=None):
    """
    Appriximate P(Y=marks | r_t^{(j)}) by sampling I^K|R.

    We use logsumexp to make this numerically more stable.

    Parameters
    ----------
    marks : array-like, with shape (n_marks, D)
        Observed marks, with shape (n_marks, D), where D is the dimensionality
        of the mark space, and n_marks is the number of observed marks.
    rates : array-like
        Relative rates of shape (n_clusters,) for current time window, and state
    cluster_means : array-like, with shape (n_clusters, D)
        D-dimensional means for each of the n_clusters neurons.
    cluster_covars : array-like, with shape (n_clusters, D, D)
        D-by-D-dimensional covariances for each of the n_clusters neurons.
    n_samples : int, optional
        Number of samples to use to approximate P(Y | S, r). Ignored
        if stype == 'no-ml'. Default is 15000.
    stype : str, optional
        One of ['unbiased', 'biased', 'max-mvn'].
        'biased' samples in proportion to mark and rate probabilities.
        'max-mvn' does not sample, but only returns the maximum likely
        IKR, based on the cluster params. Default is 'unbiased'.
    reorder : boolean, optional (default is False)
        If True, then the samples IKR will be reordered to maximize the
        likelihood. The ordering is not guaranteed to be the glabally optimal
        ordering (at least not at this time), but it is guaranteed to be some
        locally (greedy) optimal solution.

    Returns
    -------
    logP : log probability of observing sequence of marks

    """
    n_marks = len(marks)

    if n_marks == 0:
        pois = poisson(rates)
        p = pois.logpmf(0)
        logP = np.sum(p)
        return logP

    cum_rate = np.sum(rates)
    rates = rates / cum_rate

    if n_samples is None:
        n_samples = 15000

    n_clusters = len(cluster_means)

    if stype == 'unbiased':
        ikr = sample_IKR(rates=rates*cum_rate,
                     n_marks=n_marks,
                     n_samples=n_samples,
                     mode='id',
                     random_state=random_state)
    elif stype == 'biased':
        # here we bias the rates by also taking the multivariate normal cluster
        # observations into account.
        logF = np.zeros((n_clusters, n_marks)) # precomputed multivariate normal probs
        for nn in range(n_clusters):
            # multivariate normal
            mvn = multivariate_normal(mean=cluster_means[nn], cov=cluster_covars[nn])
            f = np.atleast_1d(mvn.logpdf(marks))
            # f[f < MIN_LOGLIKELIHOOD] = MIN_LOGLIKELIHOOD
            logF[nn,:] = f
        slogF = logsumexp(logF, axis=1)
        slogF = slogF - logsumexp(slogF) # normalize to a discrete prob distribution
        biased_rates = (np.exp(slogF) + rates)/2
        ikr = sample_IKR(rates=biased_rates*cum_rate,
                     n_marks=n_marks,
                     n_samples=n_samples,
                     mode='id',
                     random_state=random_state)
    elif stype == 'max-mvn':
        reorder = False
        logF = np.zeros((n_clusters, n_marks)) # precomputed multivariate normal probs
        for nn in range(n_clusters):
            # multivariate normal
            mvn = multivariate_normal(mean=cluster_means[nn], cov=cluster_covars[nn])
            f = np.atleast_1d(mvn.logpdf(marks))
            f[f < MIN_LOGLIKELIHOOD] = MIN_LOGLIKELIHOOD
            logF[nn,:] = f
        ikr = np.atleast_2d(np.argmax(logF, axis=0).astype(int))
    else:
        ikr = None
        raise ValueError("Unknown sampling type. Got '{}', expected one of []'unbiased', 'biased', 'max-mvn'].".format(stype))

    ll = eval_mark_loglikelihoods(marks=marks,
                                ikr=ikr,
                                cluster_means=cluster_means,
                                cluster_covars=cluster_covars,
                                rates=rates*cum_rate,
                                reorder=reorder)

    logP = logsumexp(ll)

    # if we used sampling, we need to normalize by number of samples:
    if stype != 'max-mvn':
        logP -= np.log(n_samples)

    return logP

def log_marked_poisson_density(X, rates, cluster_means, cluster_covars, n_samples,
                               stype='unbiased', random_state=None, reorder=False):
    """Compute the log probability under a multivariate Gaussian marked
    Poisson 'distribution'.

    Parameters
    ----------
    X : array_like, shape (n_obs, )
        List of mark features. Each element is a (n_marks, D) array, where n_marks is the
        number of marks in the particular observation, and D is the mark feature
        dimension.
    rates : array_like, shape (n_components, n_clusters)
        List of n_clusters-dimensional relative rate vectors for n_components.
        Each row corresponds to a single rate vector.
    cluster_means : array-like, with shape (n_clusters, D)
        D-dimensional means for each of the n_clusters neurons.
    cluster_covars : array-like, with shape (n_clusters, D, D)
        D-by-D-dimensional covariances for each of the n_clusters neurons.
    n_samples : int, optional
        Number of samples to use to approximate P(Y | S, r). Ignored
        if stype == 'no-ml'. Default is 15000.
    stype : str, optional
        One of ['unbiased', 'biased', 'max-mvn'].
        'biased' samples in proportion to mark and rate probabilities.
        'max-mvn' does not sample, but only returns the maximum likely
        IKR, based on the cluster params. Default is 'unbiased'.
    reorder : boolean, optional (default is False)
        If True, then the samples IKR will be reordered to maximize the
        likelihood. The ordering is not guaranteed to be the glabally optimal
        ordering (at least not at this time), but it is guaranteed to be some
        locally (greedy) optimal solution.
    Returns
    -------
    lpr : array_like, shape (n_obs, n_components)
        Array containing the log probabilities of each sample in
        X under each of the n_components mark distributions.
    """

    import multiprocessing as mp
    import psutil

    n_processes = psutil.cpu_count()

    n_components, n_clusters = rates.shape
    n_obs = len(X)
    lpr = np.zeros((n_obs, n_components))

    for zz in range(n_components):
        r = rates[zz,:].squeeze()
        with mp.Pool(processes=n_processes) as pool:
            results = [pool.apply_async(eval_P_Y_given_ISR,
                kwds={'marks' : obs,
                    'rates' : r,
                    'cluster_means': cluster_means,
                    'cluster_covars' : cluster_covars,
                    'n_samples' : n_samples,
                    'stype' : stype,
                    'random_state': random_state,
                    'reorder':reorder}) for obs in X]
            results = [p.get() for p in results]
        lpr[:,zz] = results

    lprs = np.sum(lpr, axis=1)
    lpr[lprs==0,:] = np.log(1/n_components)

    return lpr

    ############################################################################
    ############################################################################
    ############################################################################

def mp_log_marked_poisson_density(X, rates, cluster_ids, cluster_means,
                                      cluster_covars, n_samples,
                                      stype='unbiased', random_state=None,
                                      reorder=False):
    """Compute the log probability under a multivariate Gaussian marked
    Poisson 'distribution', for multiprobe (mp) data.

    Parameters
    ----------
    X : array_like, shape (n_probes, )
        List of mark features. Each element has shape (n_obs, ), where each of
        those element is a (n_marks, D) array, where n_marks is the
        number of marks in the particular observation, and D is the mark feature
        dimension.
    rates : array_like, shape (n_components, n_clusters)
        List of n_clusters-dimensional relative rate vectors for n_components.
        Each row corresponds to a single rate vector.
    cluster_means : array-like, with shape (n_probes, )-->(n_clusters, D)
        D-dimensional means for each of the n_clusters neurons.
    cluster_covars : array-like, with shape (n_probes)-->(n_clusters, D, D)
        D-by-D-dimensional covariances for each of the n_clusters neurons.
    n_samples : int, optional
        Number of samples to use to approximate P(Y | S, r). Ignored
        if stype == 'no-ml'. Default is 15000.
    stype : str, optional
        One of ['unbiased', 'biased', 'max-mvn'].
        'biased' samples in proportion to mark and rate probabilities.
        'max-mvn' does not sample, but only returns the maximum likely
        IKR, based on the cluster params. Default is 'unbiased'.
    reorder : boolean, optional (default is False)
        If True, then the samples IKR will be reordered to maximize the
        likelihood. The ordering is not guaranteed to be the glabally optimal
        ordering (at least not at this time), but it is guaranteed to be some
        locally (greedy) optimal solution.
    Returns
    -------
    lpr : array_like, shape (n_obs, n_components)
        Array containing the log probabilities of each sample in
        X under each of the n_components mark distributions.
    """

    import multiprocessing as mp
    import psutil

    n_processes = psutil.cpu_count()

    n_obs, n_probes = X.shape
    n_components, n_clusters = rates.shape
    lpr = np.zeros((n_obs, n_components))

    for zz in range(n_components):
        temp = np.zeros((n_probes, n_obs))
        for probe in range(n_probes):
            data = X[:,probe]
            r = rates[zz,cluster_ids[probe]].squeeze()
            with mp.Pool(processes=n_processes) as pool:
                results = [pool.apply_async(eval_P_Y_given_ISR,
                    kwds={'marks' : obs,
                        'rates' : r,
                        'cluster_means': cluster_means[probe],
                        'cluster_covars' : cluster_covars[probe],
                        'n_samples' : n_samples,
                        'stype' : stype,
                        'random_state': random_state,
                        'reorder':reorder}) for obs in data]
                results = [p.get() for p in results]
            temp[probe,:] = results
        temp = logsumexp(temp, axis=0)
        lpr[:,zz] = temp

    lprs = np.sum(lpr, axis=1)
    lpr[lprs==0,:] = np.log(1/n_components)

    return lpr
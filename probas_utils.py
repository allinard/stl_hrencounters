import numpy as np
from scipy import linalg
import scipy.stats as st
from pSTL import MultivariateNormalDistribution


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set of Gaussians qm,qv.
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
                  
    Takes as input 4 parameters:
        * m0: mean vector of Gaussian 1
        * S0: covariance vector of Gaussian 1
        * m1: mean vector of Gaussian 2
        * S1: covariance vector of Gaussian 2
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 


def js_divergence(p, q):
    """
        Returns the Jensen-Shannon Divergence of two MultivariateNormalDistribution
        D_{\text{JS}}(\mathcal{N}_0 \parallel \mathcal{N}_1) = \frac{1}{2} D_{\text{KL}}(\mathcal{N}_0 \parallel M) + \frac{1}{2} D_{\text{KL}}(\mathcal{N}_1 \parallel M)
        Takes as input:
            * p: a MultivariateNormalDistribution
            * q: another MultivariateNormalDistribution
    """
    s = p + q
    m = MultivariateNormalDistribution(s.mu*0.5, s.sigma*0.5)
    return 0.5 * kl_mvn(p.mu, p.sigma, m.mu, m.sigma) + 0.5 * kl_mvn(q.mu, q.sigma, m.mu, m.sigma)


def js_distance(p, q):
    """
        Returns the square-root of the Jensen-Shannon Divergence of two MultivariateNormalDistribution, which is a metric
        Takes as input:
            * p: a MultivariateNormalDistribution
            * q: another MultivariateNormalDistribution
    """
    return js_divergence(p, q) ** 0.5


def proba_datapoint_in_mvn(x, m0, S0):
    """
        Returns the probability of a data point to fit a MultivariateNormalDistribution
        \mathbb{P}(\sigma(t_i) \mid \mathcal{N}(\mu,\Sigma)) = 1 - \mathbb{P}( Q \leq (\sigma(t_i) - \mu)^T \Sigma^{-1}(\sigma(t_i) - \mu))
        Takes as input:
            * x: a data point
            * m0: the mean vector of the MultivariateNormalDistribution
            * S0: the covariance matrix of the MultivariateNormalDistribution
    """
    m_dist_x = np.dot((x-m0).transpose(),np.linalg.inv(S0))
    m_dist_x = np.dot(m_dist_x, (x-m0))
    return 1-st.chi2.cdf(m_dist_x, len(x))


import numpy as np
from scipy import linalg
import scipy.stats as st

class Conjunction:
    """
        Class representing the Conjunction operator, s.t. \phi_1 \wedge \phi_2 \wedge \ldots \wedge \phi_n.
        The constructor takes 1 arguments:
            * lst_conj: a list of pSTL formulae in the conjunction
        The class contains 1 additional attributes:
            * sat: a function \sigma(t_i) \models \phi_1 \land \phi_2 \land  \ldots \land \phi_n \Leftrightarrow (\sigma(t_i) \models \phi_1 ) \land (\sigma(t_i) \models \phi_2) \land \ldots \land (\sigma(t_i) \models \phi_n )
    """
    def __init__(self,lst_conj):
        self.lst_conj = lst_conj
        self.sat = lambda s, t : all([formula.sat(s,t) for formula in self.lst_conj])
    
    def __str__(self):
        s = "("
        for conj in self.lst_conj:
            s += str(conj) + " \wedge "
        return s[:-8]+")"


class Disjunction:
    """
        Class representing the Disjunction operator, s.t. \phi_1 \vee \phi_2 \vee \ldots \vee \phi_n.
        The constructor takes 1 arguments:
            * lst_disj: a list of pSTL formulae in the disjunction
        The class contains 1 additional attributes:
            * sat: a function \sigma(t_i) \models \phi_1 \vee \phi_2 \vee  \ldots \vee \phi_n \Leftrightarrow (\sigma(t_i) \models \phi_1 ) \vee (\sigma(t_i) \models \phi_2) \vee \ldots \vee (\sigma(t_i) \models \phi_n )
    """
    def __init__(self,lst_disj):
        self.lst_disj = lst_disj
        self.sat = lambda s, t : any([formula.sat(s,t) for formula in self.lst_disj])
    
    def __str__(self):
        s = "("
        for disj in self.lst_disj:
            s += str(disj) + " \\vee "
        return s[:-6]+")"


class Always:
    """
        Class representing the Always operator, s.t. \mathcal{G}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * proba: a probabilistic predicate of the class MultivariateNormalDistribution 
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 1 additional attributes:
            * sat: a function \sigma(t_i) \models \Box_{I} \chi^{\epsilon}   & \Leftrightarrow & \forall i' \in t_i+I\ \text{s.t.}\ \mathbb{P}(\sigma(t_{i'}) \mid \chi) \geq \epsilon
    """
    def __init__(self,t1,t2,proba):
        self.t1 = t1
        self.t2 = t2
        self.proba = proba
        self.sat = lambda s, t : all([proba.sat(s,k) for k in range(t+self.t1, t+self.t2+1)])
    
    def __str__(self):
        return "\square_{["+str(self.t1)+","+str(self.t2)+"]}\\chi_{"+str(self.proba.ID)+"}^{"+str(self.proba.epsilon)+"}"


class Eventually:
    """
        Class representing the Eventually operator, s.t. \mathcal{F}_{[t1,t2]} \phi.
        The constructor takes 3 arguments:
            * proba: a probabilistic predicate of the class MultivariateNormalDistribution 
            * t1: lower time interval bound
            * t2: upper time interval bound
        The class contains 1 additional attributes:
            * sat: a function \sigma(t_i) \models \diamondsuit_{I} \chi^{\epsilon}   & \Leftrightarrow & \exists i' \in t_i+I\ \text{s.t.}\ \mathbb{P}(\sigma(t_{i'}) \mid \chi) \geq \epsilon
    """
    def __init__(self,t1,t2,proba):
        self.t1 = t1
        self.t2 = t2
        self.proba = proba
        self.sat = lambda s, t : any([proba.sat(s,k) for k in range(t+self.t1, t+self.t2+1)])
    
    def __str__(self):
        return "\diamondsuit_{["+str(self.t1)+","+str(self.t2)+"]}\\chi_{"+str(self.proba.ID)+"}^{"+str(self.proba.epsilon)+"}"


class MultivariateNormalDistribution:
    """
        Class representing the MultivariateNormalDistribution operator \chi^{\epsilon}
        The constructor takes 4 arguments:
            * mu: the mean vector of the distribution
            * sigma: the covariance vector of the distribution
            * epsilon: probabilistic bound
            * signals: signals associated to the distribution
        The class contains 1 additional attributes:
            * sat: a function \mathbb{P}(\sigma(t_i) \mid \mathcal{N}(\mu,\Sigma)) = 1 - \mathbb{P}( Q \leq (\sigma(t_i) - \mu)^T \Sigma^{-1}(\sigma(t_i) - \mu))
    """
    
    ID_MV = 1
    
    def __init__(self,mu,sigma,epsilon=None,signals=None):
        self.mu=mu
        self.sigma=sigma
        self.epsilon=epsilon
        self.ID = MultivariateNormalDistribution.ID_MV
        self.signals = signals
        self.trajectoryIDs = []
        self.interval = None
        MultivariateNormalDistribution.ID_MV += 1
        def satfunc(s, t):
            m_dist_x = np.dot((s[t]-self.mu).transpose(),np.linalg.inv(self.sigma))
            m_dist_x = np.dot(m_dist_x, (s[t]-self.mu))
            return 1-st.chi2.cdf(m_dist_x, len(s[t])) >=  self.epsilon
        self.sat = satfunc
    
    def __add__(self,other):
        sum_mu = self.mu + other.mu
        sum_sigma = self.sigma + other.sigma
        return MultivariateNormalDistribution(sum_mu,sum_sigma)
    
    def avg(self,other):
        avg_mu = (self.mu + other.mu) * 0.5
        avg_sigma = ((((self.sigma.dot(self.sigma)) + (self.mu.dot(self.mu))) + ((other.sigma.dot(other.sigma)) + (other.mu.dot(other.mu)))) * 0.5 ) - (avg_mu.dot(avg_mu))
        return MultivariateNormalDistribution(avg_mu,avg_sigma)
    
    def __str__(self):
        return "\\chi_{"+str(self.ID)+"}^{"+str(self.epsilon)+"}"
    
    def __repr__(self):
        return str(self.ID)
    
    def bmatrix(a):
        """
        Returns a LaTeX bmatrix
        :a: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(a.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{scriptsize}\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv +=  [r'\end{bmatrix}\end{scriptsize}']
        return '\n'.join(rv)
    
    def totex(self):
        return "\chi_{"+str(self.ID)+"}^{"+str(self.epsilon)+"} = \mathcal{N}(\mu,\Sigma),\ \mu="+MultivariateNormalDistribution.bmatrix(np.round(self.mu, decimals=3))+",\ \\Sigma="+MultivariateNormalDistribution.bmatrix(np.round(self.sigma, decimals=3))


class Interval():
    """
        Class representing an Interval [t1,t2]
    """
    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2
        
    def __str__(self):
        return "("+str(self.t1)+","+str(self.t2)+")"


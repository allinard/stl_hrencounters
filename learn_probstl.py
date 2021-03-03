import sys, getopt, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 25
import scipy.stats as st
from sklearn import mixture
from itertools import groupby
from scipy import signal
import dill as pickle_dill
import pickle

from graph import Node, Graph
from probas_utils import js_distance, proba_datapoint_in_mvn
from pSTL import Conjunction, Disjunction, Always, Eventually, MultivariateNormalDistribution, Interval
from hierarchical import hierarchical_clustering_gaussians


def nearest_neighbors_ndistr(x, y):
    """
        For each normal distribution in x, finds the closest distribution in y, without reusing distributions from y. 
        The output is a 1-1 mapping of indices of distribution from x to indices of distributions from y. 
        This function takes 4 arguments:
            x: list of MultivariateNormalDistributions (at time t)
            y: list of MultivariateNormalDistributions (at time t+1)
        Returns nearest_neighbor: 1-1 mapping of distributions (from t to t+1)
    """
    x, y = map(np.asarray, (x, y))
    y = y.copy()
    y_idx = np.arange(len(y))
    nearest_neighbor = np.empty((len(x),), dtype=np.intp)
    for j, xj in enumerate(x) :
        idx = np.argmin([js_distance(xy,xj) for xy in y])
        nearest_neighbor[j] = y_idx[idx]
        y = np.delete(y, idx)
        y_idx = np.delete(y_idx, idx)

    return nearest_neighbor




def evaluate(pstl, positive_data, negative_data, verbose=False):
    """
        Function permoring evaluation of a pSTL formula against positive and negative data
        This function takes 4 arguments:
            * pstl: \phi_1
            * positive_data:
            * negative_data:
            * verbose (optional):
        Returns: ACC, BA, TPR, TNR
            * ACC: accuracy of pstl
            * BA : balanced accuracy of pstl
            * TPR: true positive rate of pstl
            * TNR: true negative rate of pstl
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    P = len(positive_data)
    N = len(negative_data)
    
    for s in positive_data:
        if pstl.sat(s,0):
            TP += 1
        else:
            FN += 1
    for s in negative_data:
        if pstl.sat(s,0):
            FP += 1
        else:
            TN += 1
    
    TPR = TP/P
    TNR = TN/N
    ACC = (TP+TN)/(P+N)
    BA  = ((TP/P) + (TN/N))/2
    
    if verbose:
        print("\n",TP,FN,FP,TN)
        print("accuracy",ACC)
        print("balanced accuracy",BA)
        print("tpr",TPR)
        print("tnr",TNR)
        
    return ACC, BA, TPR, TNR


def learn_stl(trajectories, verbose=False, alpha=1.6, beta=5, gamma=10, theta=50, w=9, p=1, H=114):
    """
        Function learning a pSTL formula from a set of trajectories/signals
        This function takes 9 arguments:
            * trajectories: the set of trajectories, in the form of a list of n-dimensional datapoints over time steps
            * alpha (optional, default 1.6): maximum distance within clusters of normal distributions at time t
            * beta (optional, default 5): maximum number of normal distributions clustering datapoints at time t
            * gamma (optional, default 10): prunning factor of possible conjunctions in the final pSTL formula (any number between 0 and 99).
            * theta (optional, default 50): tightness factor (any number between 1 and 99)
            * w (optional, default 9): Savitzky-Golay filter's window
            * p (optional, default 1): Savitzky-Golay filter's degree
            * H (optional, default H): maximum horizon of the pSTL formula
        Returns: pstl, RELEVANT_CHIS
            * pstl: a pSTL formula 
            * RELEVANT_CHIS: probabilistic predicates forming pstl
    """

    cv_type = 'full'
    
    #TODO remove this variable
    MAX_ALWAYS = H
    
    dict_time_gaussians = {}
    dict_time_gmm = {}
 
 
    #retrieving the number of normal distributions adequately describing the trajectories at time t
    if verbose:
        print("calculating normal distributions for time stamps")
    for t in range(1,H+1):
        
        #retrieve trajectory at time t
        X = []
        for trajectory in trajectories:
            X.append(trajectory[t])
        X = np.array(X)    
        
        if verbose:
            print("t=",t)
                
        #calculate how many clusters of gaussians can be retrieved within alpha using hierarchical clustering
        gmm = mixture.GaussianMixture(n_components=beta,
                                  covariance_type=cv_type, random_state=0,reg_covar=0.05)
        gmm.fit(X)
        max_gaussians = []
        for i in range(0,beta):
            max_gaussians.append(MultivariateNormalDistribution(gmm.means_[i],gmm.covariances_[i]))
        N_COMPONENTS = hierarchical_clustering_gaussians(max_gaussians, alpha)
        
        #Calculate gaussians
        gmm = mixture.GaussianMixture(n_components=N_COMPONENTS,
                                  covariance_type=cv_type, random_state=0)
        gmm.fit(X)
        pred = gmm.predict(X)
        
        gaussians = []
        for i in range(0,N_COMPONENTS):
            mnd = MultivariateNormalDistribution(gmm.means_[i],gmm.covariances_[i])
            mnd.signals = np.array([X[index] for index in [j for j, value in enumerate(pred) if value == i]])
            mnd.trajectoryIDs = np.array([index for index in [j for j, value in enumerate(pred) if value == i]])
            gaussians.append(mnd)
                
        dict_time_gaussians[t] = gaussians
        dict_time_gmm[t] = gmm
    



    #Smoothing the number of normal distributions describing the trajectories over time
    list_t_lengaussians = []
    for t in dict_time_gaussians:
        list_t_lengaussians.append(len(dict_time_gaussians[t]))
    if verbose:
        print("Smoothing the number of normal distributions over time -- creating intervals")
        print(list_t_lengaussians)
    smooth_list_t_lengaussians = [int(round(i)) for i in signal.savgol_filter(list_t_lengaussians,w,p)]   
    if verbose:
        print(smooth_list_t_lengaussians) 
    #get mismatching indexes
    mismatching_t =  [idx for idx, elem in enumerate(smooth_list_t_lengaussians) if elem != list_t_lengaussians[idx]]
    #recalculating if necessary
    for mismatching in mismatching_t:
        if verbose:
           print("\t recalculating t=", mismatching+1)
        
        X = []
        for trajectory in trajectories:
            X.append(trajectory[mismatching+1])
        X = np.array(X) 
        
        gmm = mixture.GaussianMixture(n_components=smooth_list_t_lengaussians[mismatching],
                                  covariance_type=cv_type, random_state=0)
        gmm.fit(X)
        pred = gmm.predict(X)

        
        gaussians = []
        for i in range(0,smooth_list_t_lengaussians[mismatching]):
            mnd = MultivariateNormalDistribution(gmm.means_[i],gmm.covariances_[i])
            mnd.signals = np.array([X[index] for index in [j for j, value in enumerate(pred) if value == i]])
            mnd.trajectoryIDs = np.array([index for index in [j for j, value in enumerate(pred) if value == i]])
            gaussians.append(mnd)
            
        
        dict_time_gaussians[mismatching+1] = gaussians
        dict_time_gmm[mismatching+1] = gmm
    

    
    
    #linkage of normal distributions over time
    if verbose:
        print("processing time linkage")
    t_change_nbclust = [idx for idx, elem in enumerate(smooth_list_t_lengaussians) if elem != list_t_lengaussians[idx]]
    
    time = 1
    RELEVANT_CHIS = {}
    
    CHIS_INTERVALS = {}
    
    for i,j in groupby(smooth_list_t_lengaussians):
        list_j = list(j)
        
        for chunk_chain in [list_j[i:i + MAX_ALWAYS] for i in range(0, len(list_j), MAX_ALWAYS)]:

            chain = {}        
            for nbel in range(0,i):
                chain[nbel] = []        
                gauss_t = dict_time_gaussians[time][nbel]
                chain[nbel].append(gauss_t)
                
            for t in range(time+1,time+len(chunk_chain)-1):
                list_nn = nearest_neighbors_ndistr([chain[nbel][-1] for nbel in range(0,i)],dict_time_gaussians[t]) 
                for nbel in range(0,i):
                    chain[nbel].append(dict_time_gaussians[t][list_nn[nbel]])

            
            lst_interval    = []
            for elchain in chain:
                dataset = []
                trajIDs = []
                for mnd in chain[elchain]:
                    dataset.extend(mnd.signals)
                    trajIDs.extend(mnd.trajectoryIDs)
                dataset = np.array(dataset)
                                
                #high reg_covar to force high probabilities of signals
                gmm = mixture.GaussianMixture(n_components=1,covariance_type=cv_type, random_state=0)
                gmm.fit(dataset)
                chi = MultivariateNormalDistribution(gmm.means_[0],gmm.covariances_[0],signals=dataset)
                chi.trajectoryIDs = [i for i in set(trajIDs) if trajIDs.count(i) >= len(chunk_chain)/2]
                                
                #Postcomputing the epsilon of chis
                chi.epsilon = round(np.percentile([proba_datapoint_in_mvn(x, chi.mu, chi.sigma) for x in dataset], theta),3)
                chi.interval = Interval(time,time+len(chunk_chain)-1)
                
                lst_interval.append(chi)
            
            CHIS_INTERVALS[Interval(time,time+len(chunk_chain)-1)] = lst_interval
            
            time += len(chunk_chain)


    #Do postprocessing of all chis in intervals
    #Graph based approach where all chis are vertices, and are linked 
    if verbose:
        print("building STL formulae")
    
    lst_chisintervals = list(CHIS_INTERVALS)
    
    graph = Graph()
    
    for index, obj in enumerate(lst_chisintervals):
        for node_from in CHIS_INTERVALS[obj]:
            graph.weights[node_from] = {}
            try:
                nodes_to = CHIS_INTERVALS[lst_chisintervals[index+1]]
                graph.successors[node_from] = [node_to for node_to in nodes_to]
                for node_to in nodes_to:
                    graph.weights[node_from][node_to] = len(set(node_from.trajectoryIDs)&set(node_to.trajectoryIDs))
            except IndexError:
                pass
    
    graph.remove_below_threshold(gamma)
    
    paths = []
    for nbinitials in CHIS_INTERVALS[lst_chisintervals[0]]:
        paths.extend(graph.dfs(path = [nbinitials], paths = []))
        
    pstl = Disjunction([Conjunction([Eventually(chi.interval.t1,chi.interval.t2,chi) for chi in path]) for path in paths])
    
    for path in paths:
        for chi in path:
            RELEVANT_CHIS[chi.ID] = chi
    if verbose:
        print(pstl)
    
    
    return pstl, RELEVANT_CHIS



"""
    MAIN
"""
if __name__ == '__main__':


    #Default values of parameters
    alpha=1.6
    beta=5
    gamma=10
    theta=50
    H=114
    verbose = False
    plot = False
    pickle_res = False
    
    negative_data = pickle.load( open( "user_study/data/trajectories_collision.p", "rb" ) )
    positive_data = pickle.load( open( "user_study/data/trajectories_nocollision.p", "rb" ) )
    
    #TODO CLI args for Savitzky-Golay filter parameters
    w=9
    p=1
    
    #Parse CLI parameters and replace default parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:n:a:b:g:h:t:v:d:p:",["input=","negative=","alpha=","beta","gamma=","horizon=","theta=","verbose=","dilloutput=","plot="])
    except getopt.GetoptError:
        print('some options not filled, will proceed with default parameters for these')
        
    dictbool = {'0':False, '1':True}
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            inputfile = arg
            positive_data = pickle.load( open(inputfile, "rb" ) )
        elif opt in ("-n", "--negative"):
            inputfile = arg
            negative_data = pickle.load( open(inputfile, "rb" ) )
        elif opt in ("-a", "--alpha"):
            alpha = int(arg)
        elif opt in ("-b", "--beta"):
            beta = int(arg)
        elif opt in ("-g", "--gamma"):
            gamma = int(arg)
        elif opt in ("-h", "--horizon"):
            H = int(arg)
        elif opt in ("-t", "--theta"):
            theta = int(arg)
        elif opt in ("-v", "--verbose"):
            verbose = dictbool[arg]
        elif opt in ("-d", "--dilloutput"):
            pickle_res = dictbool[arg]
        elif opt in ("-p", "--plot"):
            plot = dictbool[arg]


    #learn pSTL formula
    pstl, RELEVANT_CHIS = learn_stl(positive_data, alpha=alpha, beta=beta, gamma=gamma, theta=theta, w=w, p=p, H=H, verbose=verbose)
    
    #pickle/Dill result
    if pickle_res:
        with open("user_study/output/phi.dill", "wb") as dill_file:
            pickle_dill.dump(pstl, dill_file)
    
    #Print learnt formula
    print("\\phi = ",pstl)
    NEW_ID = 1
    for chi in RELEVANT_CHIS:
        print("\n \\\\ \n")
        gaussian = RELEVANT_CHIS[chi]
        print(gaussian.totex())
        #PLOT
        if plot:
            x, y = np.mgrid[-10:10:100j, -1:45:100j]
            pos = np.dstack((x, y))
            rv = st.multivariate_normal(gaussian.mu, gaussian.sigma)
            fig2 = plt.figure(figsize=(3,6))
            plt.tight_layout()
            ax2 = fig2.add_subplot(111)
            ax2.contourf(x, y, rv.pdf(pos))
            plt.savefig(('user_study/output/chi_'+str(gaussian.ID)+'.pdf'))
            plt.savefig(('user_study/output/chi_'+str(gaussian.ID)+'.png'))
        NEW_ID = NEW_ID+1
        

    #Evaluate against positive and negative dataset
    ACC, BA, TPR, TNR = evaluate(pstl, positive_data, negative_data, verbose=verbose)

    print("accuracy",ACC)
    print("balanced accuracy",BA)
    print("tpr",TPR)
    print("tnr",TNR)
    
        

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
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
from learn_probstl import evaluate, learn_stl


"""
    MAIN
"""
if __name__ == '__main__':

    #PARAMS
    verbose = False
    plot = False
    pickle_res = False

    
    negative_data = pickle.load( open( "user_study/data/trajectories_collision.p", "rb" ) )
    positive_data = pickle.load( open( "user_study/data/trajectories_nocollision.p", "rb" ) )
    
    
    list_thetas = []
    
    for theta in range(1,100):
    # for theta in [20]:
        pstl, RELEVANT_CHIS = learn_stl(positive_data,verbose=False,theta=theta, gamma=25)
        
        if pickle_res:
            with open("user_study/output/phi.dill", "wb") as dill_file:
                pickle_dill.dump(pstl, dill_file)
        
        if verbose:
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
            

        ACC, BA, TPR, TNR = evaluate(pstl, positive_data, negative_data, verbose=False)

        if verbose:
            print("accuracy",ACC)
            print("balanced accuracy",BA)
            print("tpr",TPR)
            print("tnr",TNR)
        
        print(theta,[TPR,TNR],BA)
        list_thetas.append([TPR,TNR])
        
    print(list_thetas)    

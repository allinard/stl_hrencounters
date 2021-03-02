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
import random
from itertools import combinations

from graph import Node, Graph
from probas_utils import js_distance, proba_datapoint_in_mvn
from pSTL import Conjunction, Disjunction, Always, Eventually, MultivariateNormalDistribution, Interval
from hierarchical import hierarchical_clustering_gaussians
from learn_probstl import evaluate, learn_stl



GAMMA = 25

def optimize_epsilon(train,negative_test,range_percentiles=range(20,52,2)):

    DICT_ID_ACC = {}
    DICT_ID_STL = {}

    id_stl = 0
    for theta in range_percentiles:
    
        pstl, _ = learn_stl(train,verbose=False,theta=theta, gamma=GAMMA)
            
        ACC, BA, TPR, TNR = evaluate(pstl, train, negative_test, verbose=False)

                
        print("\t balanced accuracy theta",theta,":",BA)
    
        DICT_ID_ACC[id_stl] = BA
        DICT_ID_STL[id_stl] = pstl
        id_stl += 1
        
        
    good_id_stl = max(DICT_ID_ACC, key=DICT_ID_ACC.get)
    return DICT_ID_STL[good_id_stl]
    
    
def split_list(l, parts):
    n = min(parts, max(len(l),1))
    k, m = divmod(len(l), n)
    return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    
    

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
    
    
    b_accuracies = []
    accuracies = []
    tpr = []
    tnr = []
    
    
    for repeating in range(0,10):
    
        
        random.shuffle(positive_data)
        random.shuffle(negative_data)
    
        cv_fold = 0
        for train_chunks in list(combinations(split_list(range(0,len(positive_data)),10),9)):
            trainidx = []
            for chunck in train_chunks:
                trainidx.extend(chunck)

            testidx = list(set(range(0,len(positive_data))) - set(trainidx))
            
            train = [positive_data[i] for i in trainidx]
            test  = [positive_data[i] for i in testidx]



            print("\nevaluating fold ",cv_fold+1)
            
            
            # pstl, RELEVANT_CHIS = learn_stl(train,verbose=False,epsilonparam=37)
            pstl = optimize_epsilon(train,negative_data)
            
            
            ACC, BA, TPR, TNR = evaluate(pstl, train, negative_data, verbose=False)

                    
            # print("\n",TP,FN,FP,TN)
            # print("accuracy",epsilonparam,(TP+TN)/(P+N))
            print("balanced accuracy fold",cv_fold+1,":",BA )
            print("accuracy fold",cv_fold+1,":", ACC )
            print("tpr fold",cv_fold+1,":", TPR)
            print("tnr fold ",cv_fold+1,":", TNR)
            b_accuracies.append(BA)
            accuracies.append(ACC)
            tpr.append(TPR)
            tnr.append(TNR)
            cv_fold += 1
        
        print("balanced accuracies",repeating)
        print(b_accuracies)
        print(np.mean(b_accuracies),np.var(b_accuracies))
        
        print("accuracies",repeating)
        print(accuracies)
        print(np.mean(accuracies),np.var(accuracies))
        
        print("tpr",repeating)
        print(tpr)
        print(np.mean(tpr),np.var(tpr))
        
        print("tnr",repeating)
        print(tnr)
        print(np.mean(tnr),np.var(tnr))   
        
        
        


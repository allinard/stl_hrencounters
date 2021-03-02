import pandas as pd
import pickle

def retrieveTrajectories():

    MAX_LENGTH = 250
    NB_TRIALS  = 20
    MIN_X_RECAL= 22.38
    Xs = []

    FILTER_LENGTH = 116
    FILTER_X      = 3.79

    LIST_PARTICIPANTS = ["2B4UDSC58D","3VQ56WB6FR","GBVEIRD1M0","M3W3JJQ52S","YZO63NDTN5","3W9J3PUCDA","FWSGS6KX8I","SCROH9LFI3","XHWHM2NPNE","DVBZOKDJFA","SA2X708V15","4RMSK8O0O7","4Y167M2N4A","OUWJHP1AUC","6TCYL5PDN8","QKAW97ORJV","YJZKD0W3RB","AD78BS3LRW","QA8KFEVTLT","MUC4LKFGIZ","A5F48AUQ2V","S3QCM9ZJ1T","YYCTRZPSVM","VKBJ8LG9IM","F7K60UI8X7","EDPAFZ8TEI","58T1SR32RD","O8AYOSGVCY","1JZL56ZWB3","S9P6VVKDJU","TNL3A5UYPU","7H6ITI4JCY","H3L61HW13D","QJC3MRAKHO","80ACJE6B5Y","34Q53LHM68","1IR22YMTFR","VGWQ9719V1","8G9Q983T4U","HTXGSZWL8X","MN1PDJ9K2D","HGFV8CTFOX","EWCVDA0R51","E41IR8CE4O","25NBIB4QZX","HGGJ7NT8WF","VMKBHJOQV9","HBB054A8FP","9TQZI1H0YB","MT8GJ6DYUB"]

    unpickled_df = pd.read_pickle("user_study/data/raw/trajectories.p")

    

    trajectories = []
    trajectories_collision = []
    trajectories_nocollision = []
    length_traj  = []

    for index, rows in unpickled_df.iterrows():

        #only retrieve experiments in determined LIST_PARTICIPANTS   
        if not unpickled_df.at[index,'completionCode'] in LIST_PARTICIPANTS:
            continue
            
            
        for trial in range(1,21):
        
            trajectory = []
            l_traj     = []
            
            for t in range(0,MAX_LENGTH):
                try:
                    x = unpickled_df.at[index,'trials_'+str(trial)+'_playerTrajectory_'+str(t)+'_0']
                    y = unpickled_df.at[index,'trials_'+str(trial)+'_playerTrajectory_'+str(t)+'_1']
                    testint = int(x)
                    trajectory.append([float(x),float(y+MIN_X_RECAL)])
                    last = [float(x),float(y+MIN_X_RECAL)]
                except KeyError:
                    l_traj.append(t)
                    trajectory.append(last)
                except ValueError:
                    l_traj.append(t)
                    trajectory.append(last)

            if min(l_traj) < FILTER_LENGTH:
                if max(abs(t[0]) for t in trajectory) < FILTER_X:
                    trajectories.append(trajectory)
                    if unpickled_df.at[index,'trials_'+str(trial)+'_collision']:
                        trajectories_collision.append(trajectory)
                    elif len(set([pos[0] for pos in trajectory])) < 3:
                        trajectories_collision.append(trajectory)
                    else:
                        trajectories_nocollision.append(trajectory)

    
    pickle.dump( trajectories_collision, open( "user_study/data/trajectories_collision.p", "wb" ) )
    pickle.dump( trajectories_nocollision, open( "user_study/data/trajectories_nocollision.p", "wb" ) )
    
    
    return trajectories, trajectories_collision, trajectories_nocollision


if __name__ == '__main__':
    _, _, _ = retrieveTrajectories()
    

# Formalizing Trajectories in Human-Robot Encounters via Probabilistic STL Inference
Dataset and code to formalize trajectories in human-robot encounters via probabilistic STL inference

## Introduction



## Downloading sources

You can use this API by cloning this repository:
```
$ git clone https://github.com/allinard/stl_hrencounters
```

Dependencies:
* Python 3.8
* numpy
* matplotlib
* scipy
* sklearn
* dill
* pandas




## Inference of probabilistic STL formulae

The module `learn_probstl.py` implements the learning probabilistic STL formulae from data.

```
pSTL = learn_stl(trajectories, 
				 alpha=1.6,
				 beta=5,
				 gamma=10,
				 theta=50,
				 w=9,
				 p=1,
				 H=114,
				 verbose=False)
```
which learns a probabilistic STL specification given a set of trajectories `trajectories`:
* `trajectories`: the set of input trajectories from which we want to learn a specification. In the form of a list of n-dimensional datapoints over time steps.
* `alpha` (optional): maximum distance within clusters of normal distributions at time t (default set to `1.6`)
* `beta` (optional): maximum number of normal distributions clustering datapoints at time t (default set to `5`)
* `gamma` (optional): prunning factor of possible conjunctions in the final pSTL formula (any number between 0 and 99, default set to `10`).
* `theta` (optional): tightness factor (any number between 1 and 99, default set to `50`)
* `w` (optional): Savitzky-Golay filter's window (default set to `9`)
* `p` (optional): Savitzky-Golay filter's degree (default set to `1`)
* `H` (optional): maximum horizon of the pSTL formula (default set to `H`)
* `verbose` (optional): print details on the execution of the algorithm (default set to `False`)




### CLI







## Dataset






## Publications

You can find hereinafter the related publications for more details on the implemented methods:
* TBD/Submitted

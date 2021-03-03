# Formalizing Trajectories in Human-Robot Encounters via Probabilistic STL Inference
Dataset and code to formalize trajectories in human-robot encounters via probabilistic STL inference


## Introduction

We are interested in formalizing human trajectories in human-robot encounters.
We consider a particular case where a human and a robot walk towards each other. A question that arises is whether, when, and how humans will deviate from their trajectory to avoid a collision. These human trajectories can then be used to generate socially acceptable robot trajectories.
To model these trajectories, we propose this data drive algorithm to extract a formal specification expressed in Signal Temporal Logic with probabilistic predicates.
We apply our method on trajectories collected through an online study where participants had to avoid colliding with a robot in a shared environment.


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

The module `learn_probstl.py` can be run in a command line:

```
python learn_probstl.py -i <input dataset> -n <input negative trajectories> -a <alpha parameter> -b <beta parameter> -g <gamma parameter> -h <horizon of formula> -t <theta parameter> -v <verbose 0/1> -d <dill pSTL formula 0/1> -p <output plot 0/1>'
```

Options are optional. By default, `-i` and `-n` refer to the dataset in `user_study/data/trajectories_nocollision` and `user_study/data/trajectories_collision` respectively.

Try, for instance:

```
python learn_probstl.py -g 25 -t 20
```






## Dataset

We also share a dataset of trajectories collected through an online study where participants had to avoid colliding with a robot in a shared environment (see [video](https://github.com/allinard/stl_hrencounters/blob/main/user_study/video.mp4)).

The raw dataset (pickle of a pandas dataframe) is located in `user_study/data/raw/trajectories.p`.

After preprocessing, it is composed of:
* `user_study/data/trajectories_nocollision`: a set of trajectories avoiding the robot properly
* `user_study/data/trajectories_collision`: a set of trajectories colliding with the robot (negative data)






## Publications

You can find hereinafter the related publications for more details on the implemented methods:
* TBD/Submitted

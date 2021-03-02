class Node():
    def __init__(self, ID, chi):
        self.ID = ID
        self.chi = chi


class Graph():
    """
        Class representing a Graph
        The constructor takes 2 (optional) arguments:
            * weights: dictionary of the form self.weights[node_from][node_to] = weight
            * successors: dictionary of the form self.successors[node_from] = [successor1, successor2 ...]
    """
    def __init__(self, weights={}, successors={}):
        self.weights = weights
        self.successors = successors

    def dfs(self, path, paths = []):
        """
            Function computing all paths from a given node
        """
        datum = path[-1]              
        if datum in self.successors:
            for val in self.successors[datum]:
                new_path = path + [val]
                paths = self.dfs(new_path, paths)
        else:
            paths += [path]
        return paths

    def __str__(self):
        s = ""
        for node_from in self.successors:
                for node_to in self.successors[node_from]:
                    s += str(node_from.ID)+" "+str(node_to.ID)+" "+str(self.weights[node_from][node_to]) + "\n"
        return s

    def remove_below_threshold(self,theta):
        """
            Function prunning the graph and removing edges not used by the dataset of trajectories 
            Takes a input `theta`: a number comprised between 0 and 100, (prunning factor)
        """
        pairs_to_del = []
        for node_from in list(self.successors):
            for node_to in list(self.successors[node_from]):
                if self.weights[node_from][node_to] < len(node_from.trajectoryIDs)* (theta/100):
                    pairs_to_del.append((node_from,node_to))
        for todel in pairs_to_del:
            del self.weights[todel[0]][todel[1]]
            self.successors[todel[0]].remove(todel[1])

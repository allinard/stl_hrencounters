import numpy as np
from scipy.cluster.hierarchy import fcluster
from probas_utils import js_distance


def calculateDistanceMatrix_gaussians_js(gaussians):
    """
    Calculate the distance matrix between n gaussians, to return an n x n condensed matrix.
    Uses the square root of the Jensen-Shannon divergence as distance.
    """
    distance_matrix = []
    i = 0
    line = []
    for f1 in gaussians:
        j = 0
        for f2 in gaussians:
            if i < j:
                line.append(js_distance(f1, f2))
            j += 1
        i += 1
    return line


def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return int( n * i - (i * (i + 1) / 2) + (j - i - 1) )
    elif i > j:
        return int( n * j - (j * (j + 1) / 2) + (i - j - 1) )


class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""

    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    def merge(self, x, y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x


def label(Z, n):
    """Correctly label clusters in unsorted dendrogram."""
    uf = LinkageUnionFind(n)
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)
        # print(Z[i, 0],Z[i, 1])


def nn_chain_gaussians(dists, n, list_gaussian_clusters, max_js_divergence_within_cluster):
    """
    linkage of gaussians
    """
    notprocessed = {el:None for el in range(n)}
    incase = [el for el in range(n)]
    alive = {el:None for el in range(n)}
    
    Z_arr = np.empty((n - 1, 4))
    Z = Z_arr

    D = dists.copy()  # Distances between clusters.
    size = np.ones(n, dtype=np.intc)  # Sizes of clusters.

    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(n, dtype=np.intc)
    chain_length = 0

    for k in range(n - 1):
        # print('lm',k,'of',n-1)
        
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break
        # Go through chain of neighbors until two mutual neighbors are found.
        while True:
            x = cluster_chain[chain_length - 1]
            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                try:
                    current_min = D[condensed_index(n, x, y)]
                except TypeError:
                    #HERE finish the linkage and return the linkage matrix
                    list_nonprocessed = list(notprocessed)
                    index = k
                    for i in range(0,len(list_nonprocessed)-1):
                        try:
                            Z_arr[index, 0] = list_nonprocessed[i]
                            Z_arr[index, 1] = list_nonprocessed[i+1]
                            Z_arr[index, 2] = MAX_DIST
                            Z_arr[index, 3] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        except IndexError:
                            break
                        size[list_nonprocessed[i+1]] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        index += 1
                    order = np.argsort(Z_arr[:, 2], kind='mergesort')
                    Z_arr = Z_arr[order]
                    label(Z_arr, n)
                    return Z_arr

            else:
                current_min = np.Infinity

                
            found = False
            for i in range(n):
            
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                
                if dist < current_min:
                    if dist<0:
                        continue
                    if dist == 0:
                        found = True
                        current_min = dist
                        y = i
                        break
                    if dist > max_js_divergence_within_cluster:
                        continue
                    found = True
                    current_min = dist
                    y = i
                    
                
            
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break
            
            if not found:
                try:
                    cluster_chain[chain_length] = incase[incase.index(x)+1]
                except IndexError:
                    
                    #HERE finish the linkage and return the linkage matrix
                    list_nonprocessed = list(notprocessed)
                    index = k
                    for i in range(0,len(list_nonprocessed)-1):
                        try:
                            Z_arr[index, 0] = list_nonprocessed[i]
                            Z_arr[index, 1] = list_nonprocessed[i+1]
                            Z_arr[index, 2] = MAX_DIST
                            Z_arr[index, 3] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        except IndexError:
                            break
                        size[list_nonprocessed[i+1]] = size[list_nonprocessed[i]] + size[list_nonprocessed[i+1]]
                        index += 1
                    order = np.argsort(Z_arr[:, 2], kind='mergesort')
                    Z_arr = Z_arr[order]
                    label(Z_arr, n)
                    return Z_arr
                    
                    
                chain_length += 1
                continue
            
            cluster_chain[chain_length] = y
            chain_length += 1
            
            
        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z[k, 0] = x
        Z[k, 1] = y
        try:
            del notprocessed[x]
            incase.remove(x)
        except KeyError:
            pass
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        del alive[x]
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster
        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue
            # D[condensed_index(n, i, y)] = js_divergence(gauss_x_y,list_gaussian_clusters[i])
            D[condensed_index(n, i, y)] = max(D[condensed_index(n, i, x)],D[condensed_index(n, i, y)])

    # Sort Z by cluster distances.
    order = np.argsort(Z_arr[:, 2], kind='mergesort')
    Z_arr = Z_arr[order]
    # Find correct cluster labels inplace.
    label(Z_arr, n)
    return Z_arr


def hierarchical_clustering_gaussians(gaussians, max_js_divergence_within_cluster):
    """
        returns how many clusters of gaussians can be retrieved within max_js_divergence_within_cluster
    """
    #Calculate distance matrix
    distance_matrix = calculateDistanceMatrix_gaussians_js(gaussians)
    # print(distance_matrix)
    #Calculate list of initial clusters
    list_gaussian_clusters = gaussians
    #Calculate full dendrogram
    linkage_matrix = nn_chain_gaussians(distance_matrix, len(gaussians), list_gaussian_clusters, max_js_divergence_within_cluster)
    # print(linkage_matrix)
    clusters = fcluster(linkage_matrix,max_js_divergence_within_cluster,'distance')
    return len(set(clusters))

import networkx as nx
import pandas as pd
import numpy as np

from active_semi_clustering.exceptions import EmptyClustersException

class AHCC:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        
    def fit(self, X, y=None, ml=[], cl=[]):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        #Construct the transitive closure of the ML constraints
        indices = [i for i in range(len(X))]
        M = self._transitive_closure(indices, ml)
        
        #If two points {x,y} are both ML and CL output "No Solution" and stop
        for cstr in ml:
            if cstr in cl or cstr[::-1] in cl:
                raise EmptyClustersException
        
        #S1 is the subset of points not involved in any must-link constraint (no need to compute)
        k_max = nx.number_connected_components(M)
        
        #Intial feasible clustering: r components M and a cluster for each point in S1
        clusters = [list(i) for i in nx.connected_components(M)]
        t = k_max
        
        centroids, distances = self._compute_centroids_distances(X, clusters)
        
        #While there is a pair of mergeable clusters (pair of clusters whose merger do not violate any CL)
        while not np.isinf(np.min(distances)):
            #Select a pair of clusters according to the specified distance criterion (centroid)
            result = np.where(distances == np.nanmin(distances))
            candidates = list(zip(result[0], result[1]))[0]

            cA, cB = clusters[candidates[0]], clusters[candidates[1]]
            
            #Check if candidate clusters are mergeable
            mergeable = True
            
            cstr = [[i,j] for i in cA for j in cB]
            for c in cstr:
                if c in cl or c[::-1] in cl:
                    distances[candidates[0]][candidates[1]] = distances[candidates[1]][candidates[0]] = np.inf
                    mergeable = False
                    break
            
            if not mergeable:
                continue
            
            #Merge candidate clusters according to specified distance criterion
            clusters.append(cA + cB) 
            clusters.remove(cA)
            clusters.remove(cB)
            
            centroids, distances = self._compute_centroids_distances(X, clusters)
            t = t - 1
            
            if self.n_clusters == t:
                break
        
        labels = np.zeros(len(X), dtype=int)
        for i in range(len(clusters)):
            for j in clusters[i]:
                labels[j] = i
        
        self.k_max_ = k_max
        self.k_min_ = t
        self.clusters_ = clusters
        self.labels_ = labels
        return self
    
    def _transitive_closure(self, indices, ml):
        M = nx.Graph()
        for i in indices:
            M.add_node(i)
        for i,j in ml:
            M.add_edge(i,j)
        return M
    
    def _compute_centroids_distances(self, X, clusters):
        centroids = [X[i].mean(axis=0) for i in clusters]
        distances = np.zeros((len(clusters),len(clusters)))

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distances[i][j] = distances[j][i] = np.linalg.norm(centroids[i] - centroids[j])

        np.fill_diagonal(distances, np.inf)
        
        return centroids, distances
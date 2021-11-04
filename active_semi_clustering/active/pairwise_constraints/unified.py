import numpy as np
import networkx as nx
import statistics
from sklearn.metrics.pairwise import euclidean_distances
from itertools import chain, zip_longest

from .example_oracle import MaximumQueriesExceeded
from .helpers import get_constraints_from_neighborhoods
from active_semi_clustering.exceptions import EmptyClustersException

class Unified:
    def __init__(self, clusterer=None, **kwargs):
        self.clusterer = clusterer
        
    def fit(self, X, oracle=None):
        neighborhoods = self._explore(X, self.clusterer.n_clusters, oracle)
        ml, cl = get_constraints_from_neighborhoods(neighborhoods)
        
        #For evaluation purposes only
        self.explore_pairwise_constraints_ = get_constraints_from_neighborhoods(neighborhoods)
        
        halt = 0        
        dist = euclidean_distances(X)
        
        #MST generation
        G = nx.Graph()
        for index in range(len(X)):
            G.add_node(index, label=-1)
        for i in G.nodes():
            for j in G.nodes():
                if i < j:
                    G.add_edge(i, j, weight=dist[i][j])
        G = nx.minimum_spanning_tree(G)
        paths = dict(nx.all_pairs_dijkstra(G))
        
        while True:
            try:
                while True:
                    try:
                        self.clusterer.fit(X, ml=ml, cl=cl)
                    except EmptyClustersException:
                        continue
                    break

                for i in G.nodes():
                    G.node[i]['label'] = self.clusterer.labels_[i]
                    
                queries_bord = {}
                for src, dst in G.edges():
                    if G.nodes[src]['label'] != G.nodes[dst]['label']:
                        queries_bord[(src,dst)] = G[src][dst]['weight']
                queries_bord = sorted(queries_bord, key=queries_bord.get)
                
                #Average size of shortest paths in MST for each node considering only nodes from their own cluster
                ds, di = {}, {}
                queries_dist = {}

                for i in paths.keys():
                    longest_path, longest_index = 0, -1
                    node_label = G.nodes[i]['label'] #Retrieve label of node i
                    node_dist = paths[i][0].copy() #All distances between node i and all nodes in the MST
                    for j in paths.keys():
                        if G.nodes[j]['label'] != node_label:
                            del node_dist[j] #Mark all nodes that do not correspond to the label of node i for removal
                        else:
                            if node_dist[j] > longest_path:
                                longest_path = node_dist[j]
                                longest_index = j
                    del node_dist[i] #Mark the node itself for removal
                    try:
                        ds[i] = statistics.mean(list(node_dist.values())) #Average SP size
                    except:
                        ds[i] = 0.0
                    di[i] = (longest_path, longest_index) #Length of longest path, index of longest path

                #If longest shortest path of a node is greater than SP average length
                #select pair for constraint generation
                for i in ds.keys():
                    if di[i][0] > np.percentile(list(ds.values()), 90):
                        queries_dist[(min(i, di[i][1]), max(i, di[i][1]))] = di[i][0]
                queries_dist = sorted(queries_dist, key=queries_dist.get, reverse=True)
                queries_dist = [i for i in queries_dist if i[0] != -1 and i[1] != -1]
                
                #Interleave two lists of different length
                queries = [x for x in chain.from_iterable(zip_longest(queries_bord, queries_dist)) if x is not None]
                
                for src, dst in queries:
                    must_linked = oracle.query(src, dst)
                    
                    if any(src in n for n in neighborhoods):
                        if any(dst in n for n in neighborhoods):
                            #Case 1 - Both elements are already in the neighborhood - DO NOTHING
                            pass
                        else:
                            #Case 2 - Only src is in the neighborhood - Find where to insert dst
                            if must_linked:
                                ml, cl, neighborhoods = self._most_informative_ml(dst, src, ml, cl, neighborhoods, oracle)
                            else:
                                ml, cl, neighborhoods = self._most_informative_cl(dst, src, ml, cl, neighborhoods, oracle)
                    else:
                        if any(dst in n for n in neighborhoods):
                            #Case 3 - Only dst is in the neighborhood - Find where to insert src
                            if must_linked:
                                ml, cl, neighborhoods = self._most_informative_ml(src, dst, ml, cl, neighborhoods, oracle)
                            else:
                                ml, cl, neighborhoods = self._most_informative_cl(src, dst, ml, cl, neighborhoods, oracle)
                        else:
                            #Case 4 - Neither element is in the neighborhood - Find where to insert them
                            if must_linked:
                                ml, cl, neighborhoods = self._most_informative_sl(src, dst, ml, cl, neighborhoods, oracle)
                            else:
                                ml, cl, neighborhoods = self._most_informative_cl(src, None, ml, cl, neighborhoods, oracle)
                                ml, cl, neighborhoods = self._most_informative_cl(dst, None, ml, cl, neighborhoods, oracle)
                                
                #Halting execution criterion (could not add any new constraint)
                if (len(cl) + len(ml)) == halt:
                    break
                else:
                    halt = len(cantLink) + len(mustLink)
                    
            except MaximumQueriesExceeded:
                break

        self.pairwise_constraints_ = ml, cl

        return self
    
    def _explore(self, X, k, oracle):
        neighborhoods = []
        traversed = []
        n = X.shape[0]

        x = np.random.choice(n)
        neighborhoods.append([x])
        traversed.append(x)

        try:
            while len(neighborhoods) < k:

                max_distance = 0
                farthest = None

                for i in range(n):
                    if i not in traversed:
                        distance = dist(i, traversed, X)
                        if distance > max_distance:
                            max_distance = distance
                            farthest = i

                new_neighborhood = True
                for neighborhood in neighborhoods:
                    if oracle.query(farthest, neighborhood[0]):
                        neighborhood.append(farthest)
                        new_neighborhood = False
                        break

                if new_neighborhood:
                    neighborhoods.append([farthest])

                traversed.append(farthest)

        except MaximumQueriesExceeded:
            pass

        return neighborhoods
    
    def _most_informative_ml(self, x_i, x_ii, ml, cl, neighborhoods, oracle):
        for neighborhood in neighborhoods:
            if x_ii in neighborhood:
                for x_j in neighborhood:
                    ml.append([x_i, x_j])

                for other_neighborhood in neighborhoods:
                    if neighborhood != other_neighborhood:
                        for x_j in other_neighborhood:
                            cl.append([x_i, x_j])

                neighborhood.append(x_i)
                break
        
        return ml, cl, neighborhoods
    
    def _most_informative_cl(self, x_i, x_ii, ml, cl, neighborhoods, oracle):
        must_link_found = False

        for neighborhood in neighborhoods:            
            if x_ii in neighborhood:
                continue

            must_linked = oracle.query(x_i, neighborhood[0])
            if must_linked:
                for x_j in neighborhood:
                    ml.append([x_i, x_j])

                for other_neighborhood in neighborhoods:
                    if neighborhood != other_neighborhood:
                        for x_j in other_neighborhood:
                            cl.append([x_i, x_j])

                neighborhood.append(x_i)
                must_link_found = True
                break

        if not must_link_found:
            for neighborhood in neighborhoods:
                for x_j in neighborhood:
                    cl.append([x_i, x_j])

            neighborhoods.append([x_i])
        
        return ml, cl, neighborhoods
    
    def _most_informative_sl(self, x_i, x_ii, ml, cl, neighborhoods, oracle):
        must_link_found = False

        for neighborhood in neighborhoods:

            must_linked = oracle.query(x_i, neighborhood[0])
            if must_linked:
                for x_j in neighborhood:
                    ml.append([x_i, x_j])
                    ml.append([x_ii, x_j])

                for other_neighborhood in neighborhoods:
                    if neighborhood != other_neighborhood:
                        for x_j in other_neighborhood:
                            cl.append([x_i, x_j])
                            cl.append([x_ii, x_j])

                neighborhood.append(x_i)
                neighborhood.append(x_ii)
                must_link_found = True
                break

        if not must_link_found:
            for neighborhood in neighborhoods:
                for x_j in neighborhood:
                    cl.append([x_i, x_j])
                    cl.append([x_ii, x_j])

            neighborhoods.append([x_i, x_ii])
        
        return ml, cl, neighborhoods

def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()
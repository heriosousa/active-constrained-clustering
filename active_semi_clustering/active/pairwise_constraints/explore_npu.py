import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .helpers import get_constraints_from_neighborhoods
from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate
from active_semi_clustering.exceptions import EmptyClustersException

class ExploreNPU(ExploreConsolidate, NPU):
    def __init__(self, clusterer=None, **kwargs):
        ExploreConsolidate.__init__(self, clusterer.n_clusters)
        NPU.__init__(self, clusterer)
    
    def fit(self, X, oracle=None):
        return ExploreConsolidate.fit(self, X, oracle)
    
    def _consolidate(self, neighborhoods, X, oracle):        
        n = X.shape[0]
        ml, cl = self.explore_pairwise_constraints_[0], self.explore_pairwise_constraints_[1]

        while True:
            try:
                while True:
                    try:
                        self.clusterer.fit(X, ml=ml, cl=cl)
                    except EmptyClustersException:
                        continue
                    break

                x_i, p_i = self._most_informative(X, self.clusterer, neighborhoods)
                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, neighborhoods)))))[1]
                must_link_found = False

                for neighborhood in sorted_neighborhoods:
                    must_linked = oracle.query(x_i, neighborhood[0])
                    if must_linked:
                        neighborhood.append(x_i)
                        must_link_found = True
                        break

                if not must_link_found:
                    neighborhoods.append([x_i])
                    
                ml, cl = get_constraints_from_neighborhoods(neighborhoods)

            except MaximumQueriesExceeded:
                break

        return neighborhoods

def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()
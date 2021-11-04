import numpy as np
import scipy as sp
import utils
from sklearn.cluster import KMeans

class CSP:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        
    def fit(self, X, y=None, ml=[], cl=[]):
        if self.n_clusters > 2:
            self.labels_ = self._constrained_spectral_clustering_K(X, ml, cl, self.n_clusters)
        else:
            self.labels_ = self._constrained_spectral_clustering(X, ml, cl)
        return self

    def _create_affinity_matrix(self, X):
        tree = spatial.KDTree(X)
        dist, idx = tree.query(X, k=16)

        idx = idx[:,1:]

        nb_data, _ = X.shape
        A = np.zeros((nb_data, nb_data))
        for i, j in zip(np.arange(nb_data), idx):
            A[i, j] = 1
        A = np.maximum(A.T, A)

        return A

    #Constraints go here ML(i,j) = Q[i][j] = 1 / CL(i,j) = Q[i][j] = -1
    def _create_constraint_matrix(self, X, must_link, cant_link):
        Q = np.zeros((len(X), len(X)), dtype=int)
        for i,j in must_link:
            Q[i][j], Q[j][i] = 1, 1
        for i,j in cant_link:
            Q[i][j], Q[j][i] = -1, -1
        Q[np.arange(Q.shape[0]), np.arange(Q.shape[0])] = 1
        return Q

    def _constrained_spectral_clustering(self, X, must_link, cant_link):
        A = self._create_affinity_matrix(X)
        Q = self._create_constraint_matrix(X, must_link, cant_link)

        D = np.diag(np.sum(A, axis=1))
        vol = np.sum(A)

        D_norm = np.linalg.inv(np.sqrt(D))
        L_norm = np.eye(*A.shape) - D_norm.dot(A.dot(D_norm))
        Q_norm = D_norm.dot(Q.dot(D_norm))

        # alpha < max eigenval of Q_norm
        alpha = 0.6 * sp.linalg.svdvals(Q_norm)[0]
        Q1 = Q_norm - alpha * np.eye(*Q_norm.shape)

        val, vec = sp.linalg.eig(L_norm, Q1)

        vec = vec[:,val >= 0]
        vec_norm = (vec / np.linalg.norm(vec, axis=0)) * np.sqrt(vol)

        costs = np.multiply(vec_norm.T.dot(L_norm), vec_norm.T).sum(axis=1)
        ids = np.where(costs > 1e-10)[0]
        min_idx = np.argmin(costs[ids])
        min_v = vec_norm[:,ids[min_idx]]

        u = D_norm.dot(min_v)

        n_dim = u.shape[0]
        p = np.zeros(n_dim)
        p[u > 0] = 1.0

        return p.astype(int)

    def _constrained_spectral_clustering_K(self, X, must_link, cant_link, K):
        A = self._create_affinity_matrix(X)
        Q = self._create_constraint_matrix(X, must_link, cant_link)

        D = np.diag(np.sum(A, axis=1))
        vol = np.sum(A)

        D_norm = np.linalg.inv(np.sqrt(D))
        L_norm = np.eye(*A.shape) - D_norm.dot(A.dot(D_norm))
        Q_norm = D_norm.dot(Q.dot(D_norm))

        # alpha < K-th eigenval of Q_norm
        alpha = 0.6 * sp.linalg.svdvals(Q_norm)[K]
        Q1 = Q_norm - alpha * np.eye(*Q_norm.shape)

        val, vec = sp.linalg.eig(L_norm, Q1)

        vec = vec[:,val >= 0]
        vec_norm = (vec / np.linalg.norm(vec, axis=0)) * np.sqrt(vol)

        costs = np.multiply(vec_norm.T.dot(L_norm), vec_norm.T).sum(axis=1)
        ids = np.where(costs > 1e-10)[0]
        min_idx = np.argsort(costs[ids])[0:K]
        min_v = vec_norm[:,ids[min_idx]]

        u = D_norm.dot(min_v) #Discretized relaxed indicator matrix

        model = KMeans(n_clusters=K).fit(u) #one of many possible discretization techniques that can derive a K-way partition from u
        labels = model.labels_

        return labels
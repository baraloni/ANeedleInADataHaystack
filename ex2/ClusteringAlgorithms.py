from sklearn.cluster import AgglomerativeClustering


def hierarchical_min_diam(data, number_of_clusters):
    clustering = AgglomerativeClustering(linkage='complete', n_clusters=number_of_clusters)
    return clustering.fit_predict(data)
  
  
def hier_cluster_by_dist(data, num_of_clusters):
    """
    Performs a hierarchical clustering.
    Assumes clusters are represented by their centroid (average).
    At each step merge the clusters with the closest centroids (i.e: euclidean distance).
    :param data: np array of shape (|data|, 2) of the data to be clustered.
    :param num_of_clusters: int in {2, 3, 4}
    :return: the clustering and a list that shows the cluster name of each point in the data
    """
    # linkage parameter is set to "ward", which minimizes the variance of the clusters being merged,
    # based on euclidean distance:
    clustering = AgglomerativeClustering(n_clusters=num_of_clusters, affinity='euclidean', linkage='ward')
    return clustering.fit_predict(data)

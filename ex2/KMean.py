import numpy as np


def l2_dist(x, y):
    """
    computes l1 distance between 2 1d numpy arrays
    """
    return np.sqrt(np.sum(np.abs(x - y) ** 2))


def recenter(cluster):
    """
    computes the center of a cluster (average of all vectors in the array)
    :param cluster: numpy array of shape (N,M) where N is the number of vectors and M is the size of each vector
    """
    tot = cluster.shape[0]
    return np.sum(cluster, axis=0) / tot


def rebase(centroids, pts):
    """
    divides the data up according to centroids provided
    :param centroids: 2D array of shape (N,M) where N is the number of vevtors and M is the size of each vector
    :param pts: 2D array of shape (K,M) where K is the number of points and M is the size of each vector
    :returns: a divison of the pts in a numpy array of shape (N,) where res[i] = index of closest centroid in centroids
    """
    res = np.zeros((pts.shape[0],))
    for i in np.arange(pts.shape[0]):
        res[i] = min_dist(pts[i], centroids)
    return res


def min_dist(x, pts):
    """
    finds the nearest vector to pts
    :param x: vector to be checked (np array of shape (M,)
    :param pts: vectors to compare to (np array of shape (N,M))
    :returns: index of nearest vector
    """
    m = l2_dist(x, pts[0])
    index = 0
    for i in np.arange(1, pts.shape[0]):
        t = l2_dist(x, pts[i])
        if t < m:
            m = t
            index = i
    return index


def choose_k_rand(k, max):
    """
    chooses k different integers in range 0 - max (assumes max > k)
    """
    rand = np.zeros((k,))
    for i in np.arange(k):
        rand[i] = np.random.randint(max)
        j = 0
        while j != i:
            if rand[i] == rand[j]:
                rand[i] = np.random.randint(max)
                j = 0
            j += 1
    return rand.astype(dtype="int16")


def k_mean(pts, k=2):
    centroids = pts[choose_k_rand(k, pts.shape[0])]
    while True:
        res = rebase(centroids, pts=pts)
        old = np.array(centroids)
        for j in np.arange(k):
            t = pts[np.where(res == j)]
            if t.shape[0] == 0:  # in case random points can generate a result with k clusters start over
                centroids = pts[choose_k_rand(k, pts.shape[0])]
                continue
            centroids[j] = recenter(t)
        if np.array_equal(old, centroids):
            break
    return res

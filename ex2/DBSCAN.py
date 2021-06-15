import numpy as np

UNCHECKED = -5
NOISE = -4
SUSPECT = -3
CURR = 0


def l2_dist(x, y):
    """
    computes l1 distance between 2 1d numpy arrays
    """
    return np.sqrt(np.sum(np.abs(x - y) ** 2))


def get_neighbors(point, pts, eps):
    '''
    finds all points which are withing epsilon of a given point
    :param point: point to be checked, np array of shape (2,) of reals
    :param pts: an np array of shape (N,2) of N points
    :param eps: supremum to distance between points
    :returns: np array of indexes in pts containing points within eps of point
    '''
    t = pts - point
    t = t ** 2
    t = np.sum(t, axis=-1)
    t = np.sqrt(t)
    res = np.where(t < eps)[0]
    return np.array(res)


def prep(pts, eps, minPts):
    '''
    chooses a random starting point for a DBSCAN cluster (i.e. an iteration of DBSCAN_loop)
    *marks down isolated points as noise
    :returns: check, div
    check is an integer state (-1 means no clusters remain in data)
    div starting point for cluster calculation
    '''
    div = np.full((pts.shape[0],), UNCHECKED)
    c = np.array(np.where(div == UNCHECKED)[0])
    while True:  # this loop creates a starting point to build a cluster
        if c.shape[0] == 0:
            return -1, div
        clpt = int(np.random.randint(c.shape[0]))
        clpt = c[clpt]
        s = get_neighbors(pts[clpt], pts, eps)
        if s.shape[0] > minPts:
            div[clpt] = CURR
            break
        else:
            div[clpt] = NOISE
        c = np.array(np.where(div == UNCHECKED)[0])
    return clpt, div


def DBSCAN_loop(pts, eps, minPts, clid, div):
    """
    computes a single cluster of id clid
    :param pts: data set of points in R^2 (np array of shape (N,2))
    :param eps: distance of point which are considered dense
    :param minPts: minimal number of point for a point to be considered core
    :param clid: cluster id
    :param div: current state of pts (with regards to clustering), this input must contain at least 1 value of 0
                otherwise the code returns div unchanged
    :returns: np array of shape (N,) containing a mapping of points in pts to either a cluster or unchecked
    ** note may override clusters existing inside div, this cannot happen unless the starting point (marked as 0) is in
    the same cluster
    """
    while True:  # this loop generates a cluster with id clid
        t = np.array(np.where(div == CURR)[0])
        check = np.array(np.where(div <= 0)[0])  # points you can check
        clst = np.array(np.where(div == clid)[0])
        if t.shape[0] == 0:
            break
        for i in t:
            s = get_neighbors(pts[i], pts[check], eps)
            s = check[s]  # shifts indexes back to those relevant to pts
            div[i] = clid
            if s.shape[0] > minPts:
                for j in s:
                    if div[j] < 0:
                        div[j] = CURR
            else:
                s2 = get_neighbors(pts[i], pts[clst], eps)
                s2 = clst[s2]
                if s.shape[0] + s2.shape[0] > minPts:
                    for j in s:
                        if div[j] < 0:
                            div[j] = CURR
                else:
                    for j in s:
                        if div[j] < 0:
                            div[j] = clid
    return div


def DBSCAN(pts, eps, minPts):
    """
    computes clusters according to DBSCAN algorithm according to input params
    :param pts: data set of points in R^2 (np array of shape (N,2))
    :param eps: distance of point which are considered dense
    :param minPts: minimal number of point for a point to be considered core
    :returns: np array of shape (N,) containing a mapping of point in pts to clusters (noise is marked in cluster -4)
    """
    res = np.full((pts.shape[0],), UNCHECKED)
    clid = 1
    while True:
        t = np.array(np.where(res == UNCHECKED)[0])
        if t.shape[0] == 0:
            return res
        clpt, div = prep(pts[t], eps, minPts)
        if clpt == -1:
            for i in np.arange(t.shape[0]):
                res[t[i]] = div[i]
            break
        div = DBSCAN_loop(pts[t], eps, minPts, clid, div)
        for i in np.arange(t.shape[0]):
            res[t[i]] = div[i]
        clid += 1
    return res








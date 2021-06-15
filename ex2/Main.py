import DBSCAN as db
import KMean as km
import numpy as np
import matplotlib.pyplot as plt
import ClusteringAlgorithms as ca
import matplotlib.backends.backend_pdf as bpdf


from DataGenerator import uniform, Gaussian_2D, Multi_Gaussian_2D, circle_in_a_ring, make_letters, Two_Moons, \
    dense_smiley_face, face


def plot(pdf, values: np.ndarray, title='fig'):
    fig = plt.figure(title)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(values.T[0], values.T[1], s=10)
    pdf.savefig(fig)
    plt.close(fig)


def plot_clustering_result(pdf, data: np.ndarray, labels: np.ndarray, title='fig'):
    fig = plt.figure(title)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(data.T[0], data.T[1], s=10)
    cluster_nums = np.unique(labels)
    for k in cluster_nums:
        indices = np.where(labels == k)
        plt.plot(data[indices].T[0], data[indices].T[1], '.')
    pdf.savefig(fig)
    plt.close(fig)


def run_clustf_k(func, pts):
    """
    runs a given clustering function on given points for number of clusters 2, 3, 4
    :param func: clustering function accepts input (data, k)
                where data is a list of points and k is the number of clusters
    :param pts: data to be processed
    :returns a list of clusterings [a, b, c] (where a is for 2 clusters, b is for 3 clusters and c is for 4 clusters]
    """
    return func(pts, 2), func(pts, 3), func(pts, 4)


def run_db_scan(pts, eps, minpts):
    """
    runs dbscan using given parameter
    :param pts: data to be processed
    :param eps: radius for dbscan algorithm (len 2 (1 for each iteration))
    :param minpts: number of point in radius for db scan (len 2 (1 for each iteration))
    :returns: a list of clusterings [a, b] where each is a run on 1 pair eps, minpts
    """
    return db.DBSCAN(pts, eps[0], minpts[0]), db.DBSCAN(pts, eps[1], minpts[1])


if __name__ == '__main__':
    pdf = bpdf.PdfPages("output.pdf")
    synthetic_data_generators = [uniform, Gaussian_2D, Multi_Gaussian_2D, circle_in_a_ring,
                                 make_letters, Two_Moons, dense_smiley_face, face]
    titles = ["Uniform distribution (x ∈ [−10, 1], y ∈ [17, 35])", "Gaussian (center = [5,1], std=3)",
              "Three Gaussians (centers= [i, −i], std=0.5 × i ∈ {1, 2, 5})", "A circle inside a ring",
              "The letters BNM",
              "Two moons", "Smiley with denser face", "Smiley with sparser face"]
    funcs = [km.k_mean, ca.hier_cluster_by_dist, ca.hierarchical_min_diam]
    eps = [[1, 2], [0.9, 0.6], [0.5, 0.5], [1, 0.8], [1.5, 0.9], [2, 3.5], [0.2, 0.2], [0.1, 0.4]]
    mins = [[6, 18], [10, 5], [8, 3], [3, 3], [2, 3], [3, 6], [6, 3], [15, 8]]
    for i in range(len(synthetic_data_generators)):
        data1 = synthetic_data_generators[i]()
        plot(pdf, data1, title=titles[i] + '\nplot 1')
        data2 = synthetic_data_generators[i]()
        plot(pdf, data2, title=titles[i] + '\nplot 2')

        a, b, c = run_clustf_k(funcs[0], data1)
        plot_clustering_result(pdf, data1, a, title=titles[i] + '\nK-means clustering, k=1')
        plot_clustering_result(pdf, data1, b, title=titles[i] + '\nK-means clustering, k=2')
        plot_clustering_result(pdf, data1, c, title=titles[i] + '\nK-means clustering, k=3')

        a, b, c = run_clustf_k(funcs[1], data1)
        plot_clustering_result(pdf, data1, a, title=titles[i] + '\nHierarchical clustering by distance, k=1')
        plot_clustering_result(pdf, data1, b, title=titles[i] + '\nHierarchical clustering by distance, k=2')
        plot_clustering_result(pdf, data1, c, title=titles[i] + '\nHierarchical clustering by distance, k=3')

        a, b, c = run_clustf_k(funcs[2], data1)
        plot_clustering_result(pdf, data1, a, title=titles[i] + '\nHierarchical clustering by diameter, k=1')
        plot_clustering_result(pdf, data1, b, title=titles[i] + '\nHierarchical clustering by diameter, k=2')
        plot_clustering_result(pdf, data1, c, title=titles[i] + '\nHierarchical clustering by diameter, k=3')

        res1, res2 = run_db_scan(data1, eps[i], mins[i])
        plot_clustering_result(pdf, data1, res1, title=titles[i] + "\nDBSCAN clustering, " + 'epsilon: ' +
                                                       str(eps[i][0]) + ', min points: ' + str(mins[i][0]))
        plot_clustering_result(pdf, data1, res2, title=titles[i] + "\nDBSCAN clustering, " + 'epsilon: ' +
                                                       str(eps[i][1]) + ', min points: ' + str(mins[i][1]))

    pdf.close()

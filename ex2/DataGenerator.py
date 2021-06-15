import numpy as np
import sklearn.datasets as ds
import sklearn.utils as util
import matplotlib.pyplot as plt


def uniform(x_low=-10.0, x_high=1.0, y_low=17.0, y_high=35.0, samples=300):
    """
    generates samples in a uniform distribution.
    in defaulting manner: 300 samples where x ∈ [−10, 1], y ∈ [17, 35].
    :param x_low: the low bound of the x value
    :param x_high: the high bound of the x value
    :param y_low: the low bound of the y value
    :param y_high: the high bound of the y value
    :param samples: integer number of samples to output
    :return: np array of shape (samples, 2) of the samples
    """
    x = np.random.uniform(x_low, x_high, samples)
    y = np.random.uniform(y_low, y_high, samples)
    return np.column_stack((x, y))


def Gaussian_2D(centers=np.array([5,1]), std=np.full((2,), 3), samples=300):
    '''
    generates samples through a Gaussian filter in R^2
    :param centers: np array of shape (2,) center of the distribution
    :param std: the standard deviation of the distribution np array of shape (2,)
    :param samples: integer number of samples to output
    :returns: np array of shape (samples, 2) of the samples
    '''
    x = np.random.normal(centers[0], std[0], samples)
    y = np.random.normal(centers[1], std[1], samples)
    return np.stack((x, y), axis=-1)


def Multi_Gaussian_2D(centers=np.array([[1, -1], [2, -2], [5, -5]]),
                      stds=np.array([[0.5, 1], [0.5, 2], [0.5, 5]]), samples=300):
    '''
    generates samples through a filter of 3 Gaussians in R^2
    ** note- if you want to use default value of centers then the shape of stds has to be (3,2) similarsly
             if you want to use the default of stds then the shape of centers has to be (3,2)
    :param centers: np array of shape (3,2) of centers for the 3 Gaussians
    :param stds: the covariance of the distribution of each Gaussian
                (indexed the same as centers) np array of shape (3,2)
    :param samples: integer number of samples to output
    :returns: np array of shape (samples, 2) of the samples
    '''
    s = np.random.randint(0, centers.shape[0], samples)
    t, counts = np.unique(s, return_counts=True)
    res = Gaussian_2D(centers[0], stds[0], counts[0])
    for i in np.arange(1, centers.shape[0]):
        res = np.concatenate([res, Gaussian_2D(centers[i], stds[i], counts[i])])
    return res


def doughnut(outer_radius, inner_radius, samples):
    """
    generate samples to create a doughnut shape.
    :param outer_radius: the radius of the outer circle.
    :param inner_radius: the radius of the inner circle.
    :param samples: integer number of samples to output.
    :returns: np array of shape (samples, 2) of the samples.
    """
    theta = 2 * np.pi * np.random.uniform(0, 1, samples)
    r = (outer_radius - inner_radius) * np.sqrt(np.random.uniform(0, 1, samples)) + inner_radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def circle_in_a_ring(samples=300, big_r=5, medium_r=4, small_r=3):
    """
    generate samples to create a circle in a ring.
    :param samples: integer number of samples to output.
    :param big_r: the radius of the outer circle of the ring.
    :param medium_r: the radius of the inner circle of the ring.
    :param small_r: the radius of the circle.
    :returns: np array of shape (samples, 2) of the samples.
    """
    # because this is a pythagorean triple it holds that ringArea = circleArea
    # so we can assign (samples / 2) for each shapes and therefore guaranty uniform distribution between the 2 shapes.
    ring = doughnut(big_r, medium_r, samples // 2)
    circle = doughnut(small_r, 0, samples - samples // 2)
    return np.concatenate((ring, circle), 0)


def dense_smiley_face(samples=3000, feature_num=200):
    """
    generate samples to create a smiley face, whose face is dence but featutes (eyes, nose and smile)
    are sparse.
    :param samples: integer number of samples to output.
    :param feature_num: the number of features samples, should be relatively low in order to
                  display the contrast between the features to the rest of the face.
    :returns: np array of shape (samples, 2) of the samples.
    """

    def mask(x, y):
        """
        check whether a sample is a feature or not.
        :param x: the x value of the sample.
        :param y: the y value or the sample.
        :return: true is the sample is a feature, false otherwise.
        """
        r = 1.5
        l_eye = (x + 2) ** 2 + (y - 1.7) ** 2 < r
        r_eye = (x - 2) ** 2 + (y - 1.7) ** 2 < r
        nose = x ** 2 + y ** 2 < 0.5 * r
        smile = y < 1 and x ** 2 / 3 + (y + 2) ** 2 < 3 * r < x ** 2 / 3 + (y + 1) ** 2
        return l_eye or r_eye or nose or smile

    smiley = np.array([[]]).reshape(0, 2)
    feature_cnt = 0
    while len(smiley) < samples:
        candidates = doughnut(5, 0, samples - len(smiley))
        for candidate in candidates:
            # candidate is inside the mask, and we don't want to cross the limit (we want the mask to be sparse):
            if feature_cnt < feature_num and mask(candidate[0], candidate[1]):
                feature_cnt += 1
                smiley = np.concatenate((smiley, [candidate]), 0)
            # candidate is outside the mask:
            if not mask(candidate[0], candidate[1]):
                smiley = np.concatenate((smiley, [candidate]), 0)
    return smiley


def Two_Moons(samples=300):
    '''
    generates samples in the shape of 2 moons (where the upper moon is more sparse)
    :param samples: number of samples to generate
    :returns: np array of shape (samples, 2) of the samples
    '''
    gen = util.check_random_state(None)
    s, s2 = ds.make_moons(samples, shuffle=False)
    t = np.where(s2 == 1)[0][0]
    a = s[:t, :]
    b = s[t:, :]
    a += gen.normal(scale=0.1, size=a.shape)
    b += gen.normal(scale=0.05, size=b.shape)
    return np.row_stack((a, b)) * 20


def make_empty_circ(size, x_radiant, y_radiant, x_indentation=2.0, y_indentation=9.0, y_strech=10.0, x_strech=0.02):
    xs = x_strech * np.sin(np.arange(0, x_radiant, x_radiant / size)) + x_indentation
    ys = y_strech * np.cos(np.arange(0, y_radiant, y_radiant / size)) + y_indentation
    return np.vstack([xs, ys])


def make_linear(size, slope, indentation, y_strech=1.0):
    xs = (slope * np.arange(0, size) / size) + indentation
    ys = np.arange(0, size) * y_strech
    return np.vstack([xs, ys])


def make_line(size, indentation, horizontal=False, y_strech=1.0):
    xs = indentation + np.zeros(size)
    ys = np.arange(0, size) * y_strech
    line = np.vstack([xs, ys])
    if horizontal:
        return line.T
    return line


def draw_b(size=100, noise_rate=0.1):
    left_line = make_line(1 + size // 3, 1)
    first_circ = make_empty_circ(size // 3, np.pi, np.pi, x_indentation=1, y_indentation=25, y_strech=8.4, x_strech=3)
    second_circ = make_empty_circ(size // 3, np.pi, np.pi, x_indentation=1, y_indentation=8, y_strech=7.5, x_strech=4)

    B = np.hstack([left_line, first_circ, second_circ])
    return B + noise_rate * np.random.normal(size=B.shape)


def make_n(size=100, noise_rate=0.1):
    line1 = make_line(size // 3, 6)
    linear_line = make_linear(1 + size // 3, -4, 10)
    line3 = make_line(size // 3, 10)
    N = np.hstack([line1, linear_line, line3])
    return N + noise_rate * np.random.normal(size=N.shape)


def make_m(size=100, noise_rate=0.1):
    line1 = make_line(size // 4, 12, y_strech=1.35)
    left_slope = make_linear(size // 4, -2, 14, 1.35)
    right_slope = make_linear(size // 4, 2, 14, 1.35)
    line3 = make_line(size // 4, 16, y_strech=1.35)
    M = np.hstack([line1, left_slope, right_slope, line3])
    return M + noise_rate * np.random.normal(size=M.shape)


def make_letters():
    N = make_n()
    M = make_m()
    B = draw_b()
    return np.hstack([N, M, B]).T


def make_full_circ(size, x_indentation=0.0, y_indentation=0.0, stretch_factor=1.0):
    r = np.random.rand(size)
    theta = 2 * np.pi * np.random.rand(size)
    xs = stretch_factor * np.sqrt(r) * np.cos(theta) + x_indentation
    ys = stretch_factor * np.sqrt(r) * np.sin(theta) + y_indentation
    return np.vstack([xs, ys])


def make_smile(size, radius, x_indentation=0.0, y_indentation=0.0, stretch_factor=1.0, noise_rate=0.04):
    theta = np.pi * np.random.rand(size)
    xs = -stretch_factor * np.sqrt(radius) * np.cos(theta) + x_indentation
    ys = -stretch_factor * np.sqrt(radius) * np.sin(theta) + y_indentation
    smile = np.vstack([xs, ys])
    return smile + noise_rate * np.random.normal(size=smile.shape)


def face():
    face = make_full_circ(100)
    eye1 = make_full_circ(30, 0.40, 0.40, stretch_factor=0.1)
    eye2 = make_full_circ(30, -0.40, 0.40, stretch_factor=0.1)
    nose = make_full_circ(30, 0, 0, stretch_factor=0.1)
    smile = make_smile(110, 0.5, 0, -0.25, 0.5)
    return np.hstack([face, eye1, eye2, nose, smile]).T

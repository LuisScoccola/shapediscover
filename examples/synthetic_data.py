import numpy as np


def add_gaussian_noise(pointcloud, variance, seed=0):
    np.random.seed(seed)
    n_points = pointcloud.shape[0]
    dimensions = pointcloud.shape[1]
    if variance:
        return pointcloud + np.random.multivariate_normal(
            mean=np.zeros(dimensions),
            cov=np.identity(dimensions) * variance,
            size=n_points,
        )
    else:
        return pointcloud


def sphere(n, d, noise_variance=0, seed=0):
    np.random.seed(seed)
    output = np.random.multivariate_normal(
        mean=np.zeros(d + 1), cov=np.identity(d + 1), size=n
    )
    output = output / np.linalg.norm(output, axis=1).reshape(-1, 1)
    return add_gaussian_noise(output, noise_variance)


def double_circle(n, noise_variance=0, seed=0):
    np.random.seed(seed)
    return np.vstack(
        (
            sphere(n, 1, noise_variance=noise_variance),
            (sphere(n, 1, noise_variance=noise_variance) + np.array([3, 0])),
        )
    )


def disk(n, d, noise_variance=0, seed=0):
    np.random.seed(seed)
    Y = sphere(n, d - 1)
    radii = np.power(np.random.random(n), 1 / d)
    output = radii.reshape(-1, 1) * Y
    return add_gaussian_noise(output, noise_variance)


def interval(n, noise_variance=0, seed=0):
    np.random.seed(seed)
    in_one_dimension = disk(n, 1, noise_variance=0)
    in_two_dimensions = np.hstack((in_one_dimension, np.zeros_like(in_one_dimension)))
    return add_gaussian_noise(in_two_dimensions, noise_variance)


def torus(N, r1=1, r2=0.5, noise_variance=0, seed=0):
    np.random.seed(seed)
    T1 = np.random.random(N)
    T2 = np.random.random(N)
    Y = np.hstack([T1.reshape(-1, 1), T2.reshape(-1, 1)])
    Y *= 2 * np.pi
    X = []
    for y in Y:
        t1, t2 = y
        pt_x = (r1 + r2 * np.cos(t2)) * np.cos(t1)
        pt_y = (r1 + r2 * np.cos(t2)) * np.sin(t1)
        pt_z = r2 * np.sin(t2)
        pt = np.array([pt_x, pt_y, pt_z])
        X.append(pt)
    X = np.array(X)
    return add_gaussian_noise(X, noise_variance)

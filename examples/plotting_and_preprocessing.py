import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import umap
import glasbey

import kmapper
from shapediscover import plot_nerve
from sklearn.cluster import DBSCAN

from shapediscover import plot_pointcloud_with_function, plot_nerve


def plot_and_return_2d_embedding(
    X,
    X_dim_red=None,
    X_dim_red_name=None,
    y=None,
    random_state=0,
    n_neighbors=15,
    point_size=1,
    legend_location="best",
    dummy_legend=False,
):
    X_umap = umap.UMAP(
        random_state=random_state, n_neighbors=n_neighbors, n_components=2
    ).fit_transform(X)
    X_pca = PCA(random_state=random_state, n_components=2).fit_transform(X)

    to_plot = [X_umap, X_pca]
    titles = ["UMAP", "PCA"]
    if X_dim_red is not None:
        to_plot.append(X_dim_red)
        if X_dim_red_name is None:
            X_dim_red_name = "input dim red"
        titles.append(X_dim_red_name)

    fig, axs = plt.subplots(1, len(to_plot))
    fig.set_figheight(7)
    fig.set_figwidth(15)

    if y is not None:
        label_encoder = LabelEncoder()
        y_numerical = np.array(label_encoder.fit_transform(y))
        classes = (
            label_encoder.classes_
            if not dummy_legend
            else list(range(len(label_encoder.classes_)))
        )

        if -1 in y:
            colors = ["grey"] + glasbey.create_palette(palette_size=len(classes)-1)
        else:
            colors = glasbey.create_palette(palette_size=len(classes))
        colormap = ListedColormap(colors)

        for two_dim_pointcloud, title, ax in zip(to_plot, titles, axs):

            ax.set_title(title)
            ax.scatter(
                two_dim_pointcloud[:, 0],
                two_dim_pointcloud[:, 1],
                c=y_numerical,
                cmap=colormap,
                s=point_size,
            )
            for color, label in zip(colors, classes):
                ax.scatter([], [], color=color, label=label)
            ax.legend(loc=legend_location)

            # ax.legend(
            #    handles=scatter.legend_elements()[0],
            #    labels=list(label_encoder.classes_),
            #    loc=legend_location,
            # )
            ax.set_aspect("equal", "box")

    else:
        for two_dim_pointcloud, title, ax in zip(to_plot, titles, axs):

            ax.set_title(title)
            ax.scatter(
                two_dim_pointcloud[:, 0],
                two_dim_pointcloud[:, 1],
                s=point_size,
            )

    return X_umap, X_pca


def plot_mapper(
    X,
    filtering_function,
    overlap_perc,
    n_intervals,
    dbscan_eps,
    X_proj,
    max_vertex_size=0.05,
    plot_letters=False,
    plot_subfunctions=["cover", "nerve"],
    plot_name=None,
):
    n_points = X.shape[0]

    mapper = kmapper.KeplerMapper(verbose=0)
    cover = dict(
        mapper.map(
            filtering_function,
            X,
            cover=kmapper.Cover(n_cubes=n_intervals, perc_overlap=overlap_perc),
            clusterer=DBSCAN(eps=dbscan_eps),
        )["nodes"]
    )
    n_cover_elements = len(cover)

    cover_as_function = np.zeros((n_cover_elements, n_points))
    for i, cover_element in enumerate(list(cover.values())):
        for j in cover_element:
            cover_as_function[i, j] = 1

    if "cover" in plot_subfunctions:
        if plot_name:
            plot_name_cover = plot_name + "_cover"
        else:
            plot_name_cover = None
        plot_pointcloud_with_function(
            X_proj,
            cover_as_function,
            point_size=1,
            enumerate_plots=True,
            plot_name=plot_name_cover,
        )
        plt.show()

    if "nerve" in plot_subfunctions:
        if plot_name:
            plot_name_nerve = plot_name + "_nerve"
        else:
            plot_name_nerve = None
        plot_nerve(
            cover_as_function,
            0.5,
            labels=np.zeros(n_points),
            max_vertex_size=max_vertex_size,
            plot_letters=plot_letters,
            interactive=False,
            plot_name=plot_name_nerve,
        )
        plt.show()


def plot_ball_mapper(
    X,
    eps,
    X_proj,
    y=None,
    max_subsample=80,
    max_vertex_size=0.05,
    interactive=False,
    plot_letters=False,
    plot_subfunctions=["nerve"],
    dummy_legend=False,
    plot_name=None,
):
    n_points = X.shape[0]
    dist_mat = sp.spatial.distance_matrix(X, X)
    ds = np.zeros(n_points)
    idx_perm = np.zeros(n_points, dtype=int)
    radii = np.zeros(n_points)
    representatives = np.zeros(n_points, dtype=int)

    idx = 0
    ds = np.copy(dist_mat[idx])

    current_eps = np.inf
    i = 0
    while current_eps > eps:
        idx = np.argmax(ds)
        idx_perm[i] = idx
        current_eps = ds[idx]
        radii[i - 1] = current_eps
        for j in range(0, n_points):
            val1 = ds[j]
            val2 = dist_mat[idx, j]
            if val1 > val2:
                ds[j] = val2
                representatives[j] = i
        i += 1
        if i >= max_subsample:
            break

    radii[-1] = np.max(ds)

    cover_size = i
    print("cover size:",cover_size)

    idx_perm = idx_perm[:cover_size]

    cover_as_function = np.zeros((cover_size, n_points))

    for j in range(n_points):
        for k_idx, k in enumerate(idx_perm):
            if dist_mat[j, k] < eps:
                cover_as_function[k_idx, j] = 1

    if "cover" in plot_subfunctions:
        if plot_name:
            plot_name_cover = plot_name + "_cover"
        else:
            plot_name_cover = None
        plot_pointcloud_with_function(
            X_proj,
            cover_as_function,
            point_size=1,
            enumerate_plots=True,
            plot_name=plot_name_cover,
        )
        plt.show()


    if "nerve" in plot_subfunctions:
        if plot_name:
            plot_name_nerve = plot_name + "_nerve"
        else:
            plot_name_nerve = None
        plot_nerve(
            cover_as_function,
            0.5,
            labels=y,
            max_vertex_size=max_vertex_size,
            plot_letters=plot_letters,
            interactive=interactive,
            dummy_legend=dummy_legend,
            plot_name=plot_name_nerve,
        )
        plt.show()


    return cover_as_function, idx_perm

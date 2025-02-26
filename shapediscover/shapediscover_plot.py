import tempfile
import numpy as np
import matplotlib.pyplot as plt
import itertools
import gudhi
import glasbey
from sklearn.preprocessing import LabelEncoder
from pyvis.network import Network

from .fuzzy_cover import (
    fuzzy_cover_to_filtered_complex,
    threshold_fuzzy_cover,
    fuzzy_cover_with_labels_to_fractions,
    fuzzy_cover_to_weighted_edges,
)
from .nerve_layout import nerve_layout


def ith_letter(i):
    n_letters = 26
    if i <= n_letters - 1:
        return chr(ord("A") + i)
    else:
        return chr(ord("a") + i - n_letters)


def shapediscover_plot(
    shapediscover,
    X,
    y=None,
    cover_threshold=0.5,
    plot_subfunctions=None,
    dimension_simplicial_complex=2,
    figsize=(15, 4),
    point_size=1,
    max_vertex_size=0.1,
    interactive=False,
    dummy_legend=False,
    seed=0,
    plot_name=None,
):

    # Function for plotting multiple things from the pipeline output
    # Run things on plot_subfunctions; default is all
    if plot_subfunctions is None:
        plot_subfunctions = [
            "initialization_precover",
            "precover",
            "fuzzy_cover",
            "cover",
            "nerve",
            "barcode",
        ]

    # Initialization precover
    if "initialization_precover" in plot_subfunctions:
        if shapediscover.initialization_precover_ is None:
            raise Exception("Must fit the ShapeDiscover object.")

        plot_pointcloud_with_function(
            X,
            shapediscover.initialization_precover_,
            point_size=point_size,
            figsize=figsize,
        )
        print("initialization precover")
        plt.show()

    # Precover
    if "precover" in plot_subfunctions:
        if shapediscover.precover_ is None:
            raise Exception("Must fit the ShapeDiscover object.")

        plot_pointcloud_with_function(
            X,
            shapediscover.precover_,
            point_size=point_size,
            figsize=figsize,
        )
        print("output precover")
        plt.show()

    # Fuzzy Cover
    if "fuzzy_cover" in plot_subfunctions:
        if shapediscover.cover_ is None:
            raise Exception("Must fit the ShapeDiscover object.")

        plot_pointcloud_with_function(
            X,
            shapediscover.cover_,
            point_size=point_size,
            figsize=figsize,
        )
        print("output fuzzy cover")
        plt.show()

    # Cover
    if "cover" in plot_subfunctions:
        if shapediscover.cover_ is None:
            raise Exception("Must fit the ShapeDiscover object.")

        thresholded_fuzzy_cover = threshold_fuzzy_cover(
            shapediscover.cover_, cover_threshold
        )[0]

        if plot_name:
            plot_name_cover = plot_name + "_cover"
        else:
            plot_name_cover = None
        plot_pointcloud_with_function(
            X,
            thresholded_fuzzy_cover,
            point_size=point_size,
            figsize=figsize,
            enumerate_plots=True,
            plot_name=plot_name_cover,
        )
        # for i, size in enumerate(np.sum(thresholded_fuzzy_cover, axis=1)):
        #    print(ith_letter(i), size)
        print("output cover")
        plt.show()

    # Nerve
    if "nerve" in plot_subfunctions:
        if shapediscover.cover_ is None:
            raise Exception("Must fit the ShapeDiscover object.")

        if plot_name:
            plot_name_nerve = plot_name + "_nerve"
        else:
            plot_name_nerve = None

        plot_nerve(
            shapediscover.cover_,
            cover_threshold,
            labels=y,
            max_dimension=dimension_simplicial_complex,
            max_vertex_size=max_vertex_size,
            seed=seed,
            dummy_legend=dummy_legend,
            interactive=interactive,
            plot_name=plot_name_nerve,
        )
        print("nerve of fuzzy cover by thresholding at", cover_threshold)
        plt.show()

    # Persistence barcode
    if "barcode" in plot_subfunctions:
        if shapediscover.gudhi_persistence_diagram_ is None:
            raise Exception(
                "Must call fit_persistence() on the ShapeDiscover object first."
            )

        print("barcode of fuzzy cover")
        if plot_name:
            plot_name_pd = plot_name + "_pd"
        else:
            plot_name_pd = None
        plot_persistence_barcode(
            shapediscover.gudhi_persistence_diagram_, plot_name=plot_name_pd
        )
        plt.show()


def plot_losses(historical_losses, losses_names, from_onwards=0, figsize=(10, 2)):
    fig = plt.figure(figsize=figsize)
    for historical_loss, loss_name in zip(historical_losses, losses_names):
        historical_loss = np.array(historical_loss)
        if historical_loss.shape[0] == 0:
            continue
        keep = historical_loss[:, 0] >= from_onwards
        plt.plot(
            historical_loss[keep, 0],
            historical_loss[keep, 1],
            marker="o",
            label=loss_name,
            markersize=0,
        )
    plt.legend()
    return fig


def plot_pointcloud_with_function(
    pointcloud,
    covers,
    point_size=10,
    figsize=(15, 8),
    enumerate_plots=False,
    plot_name=None,
):
    figsize = (10, 6)
    n_cover_elements = covers.shape[0]
    max_per_row = 6
    if n_cover_elements <= max_per_row:
        fig, axs = plt.subplots(1, n_cover_elements, figsize=figsize)
    else:
        n_rows = n_cover_elements // max_per_row
        fig, axs = plt.subplots(n_rows + 1, max_per_row, figsize=figsize)
        axs = [ax for row_of_axs in axs for ax in row_of_axs]

    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for i, ax in enumerate(axs):
        if i < n_cover_elements:
            ax.scatter(
                pointcloud[:, 0],
                pointcloud[:, 1],
                c=covers[i],
                s=point_size,
                vmin=0,
                vmax=1,
            )
            if enumerate_plots:
                ax.title.set_text(ith_letter(i))
        else:
            ax.axis("off")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect("equal")

    fig.tight_layout()
    if plot_name:
        plt.savefig(
            plot_name + ".png",
            format="png",
            bbox_inches="tight",
            dpi=150,
        )

    return fig


def scatter_letters(positions, letters, ax):
    for pos, letter in zip(positions, letters):
        ax.annotate(letter, pos)


def plot_2d_simplicial_complex(
    simplices, vertex_radii, positions, plot_vertices=True, ax=None
):
    # adapted from: https://github.com/iaciac/py-draw-simplicial-complex/blob/970953b661e1da5fe3edcb3cdfa52ce805f6fa3d/Draw%202d%20simplicial%20complex.ipynb
    if not ax:
        ax = plt.gca()

    # 1-simplices
    if len(simplices) >= 2:
        edges = list(
            set(
                itertools.chain(
                    *[
                        [
                            tuple(sorted((i, j)))
                            for i, j in itertools.combinations(simplex, 2)
                        ]
                        for simplex in simplices[1]
                    ]
                )
            )
        )

        for i, j in edges:
            (x0, y0) = positions[i]
            (x1, y1) = positions[j]
            line = plt.Line2D([x0, x1], [y0, y1], color="black", zorder=1, lw=0.7)
            ax.add_line(line)

    # 2-simplices
    if len(simplices) >= 3:
        triangles = list(
            set(
                itertools.chain(
                    *[
                        [
                            tuple(sorted((i, j, k)))
                            for i, j, k in itertools.combinations(simplex, 3)
                        ]
                        for simplex in simplices[2]
                    ]
                )
            )
        )

        for i, j, k in triangles:
            (x0, y0) = positions[i]
            (x1, y1) = positions[j]
            (x2, y2) = positions[k]
            tri = plt.Polygon(
                [[x0, y0], [x1, y1], [x2, y2]],
                edgecolor="black",
                facecolor=plt.cm.Blues(0.6),
                zorder=2,
                alpha=0.4,
                lw=0.5,
            )
            ax.add_patch(tri)

    # draw 0-simplices
    if plot_vertices:
        for pos, radius in zip(positions, vertex_radii):
            circ = plt.Circle(
                pos,
                radius=radius,
                zorder=3,
                lw=1,
                edgecolor="Black",
                facecolor="blue",
            )
            ax.add_patch(circ)


def plot_nerve(
    cover,
    threshold,
    labels=None,
    max_dimension=2,
    max_vertex_size=1,
    seed=0,
    plot_letters=True,
    dummy_legend=False,
    interactive=False,
    plot_name=None,
):

    positions, thresholded_fuzzy_cover = nerve_layout(cover, threshold, seed=seed)

    # compute higher dimensional nerve to plot higher dimensional simplices
    filtered_simplicial_complex = fuzzy_cover_to_filtered_complex(
        thresholded_fuzzy_cover, max_dimension=max_dimension
    )
    middle_of_scale = 0.5  # used only in fuzzy cover taking binay values {0,1}
    simplicial_complex = filtered_simplicial_complex.cut(middle_of_scale)

    vertex_sizes = np.sum(thresholded_fuzzy_cover, axis=1)
    # sqrt_sizes = np.sqrt(cover_sizes)
    # vertex_radii = sqrt_sizes / np.max(sqrt_sizes)

    if labels is not None:
        fractions = fuzzy_cover_with_labels_to_fractions(
            thresholded_fuzzy_cover, labels
        )
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        classes = (
            label_encoder.classes_
            if not dummy_legend
            else list(range(len(label_encoder.classes_)))
        )
    else:
        fractions = None
        classes = None

    # compute edge weights
    at_least_one_point = 1
    edges, intersection_sizes = fuzzy_cover_to_weighted_edges(
        thresholded_fuzzy_cover, min_weight=at_least_one_point
    )

    intersection_sizes = np.array(intersection_sizes)

    edge_weights = intersection_sizes
    vertex_weights = vertex_sizes

    n_vertices = len(vertex_weights)
    # radius of node is propotional to log of cover element size
    radii = np.log(vertex_weights + 1)
    normalized_radii = radii / np.max(radii)

    # width of edge is proportional to log of intersection size (scaled by *radius* of largest node)
    edge_widths = np.log(edge_weights + 1)
    normalized_edge_widths = edge_widths / np.max(radii)

    if len(edge_weights) > 0:
        edges = simplicial_complex[1]
    else:
        edges = []

    if interactive:
        if -1 in labels:
            colors = ["grey"] + glasbey.create_palette(palette_size=len(classes) - 1)
        else:
            colors = glasbey.create_palette(palette_size=len(classes))

        def _save_piechart(frac):
            _, ax = plt.subplots()
            ax.pie(frac, colors=colors)
            circ = plt.Circle(
                [0, 0],
                radius=1,
                zorder=3,
                lw=5,
                edgecolor="Black",
                fill=False,
            )
            ax.add_patch(circ)
            ax.axis("equal")

            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_file:
                plt.savefig(
                    temp_file, format="svg", transparent=True, bbox_inches="tight"
                )
                temp_file_path = temp_file.name
            plt.close()
            return temp_file_path

        def _save_plot_legend(classes, colors):
            _, ax = plt.subplots(figsize=(1, 1))
            for color, label in zip(colors, classes):
                ax.scatter([], [], color=color, label=label)
            ax.legend(loc="center")
            ax.axis("equal")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.axis("off")

            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as temp_file:
                plt.savefig(
                    temp_file, format="svg", transparent=True, bbox_inches="tight"
                )
                temp_file_path = temp_file.name
            plt.close()
            return temp_file_path

        n_vertices = len(vertex_weights)

        # edges must be a list of tuples of ints
        edges = [(int(i), int(j)) for i, j in edges]

        vertex_labels = [ith_letter(i) for i in range(n_vertices)]

        radius_scale = 60
        edge_width_scale = 10
        edge_length_scale = 0.1
        legend_scale = 60
        edges_smooth = True
        filter_menu = False
        show_physics = True
        other_buttons = False
        plot_width = "100%"
        plot_height = "1200px"
        neighborhood_highlight = False
        labelHighlightBold = False
        masses = np.full_like(np.log(vertex_weights), 1)

        radii_in_plot = radius_scale * normalized_radii

        lenghts = 1 / edge_weights
        # lenghts = np.exp(-edge_weights)
        normalized_edge_lengths = lenghts / np.max(lenghts)
        edge_lengths_in_plot = edge_length_scale * normalized_edge_lengths

        edge_widths_in_plot = edge_width_scale * normalized_edge_widths

        pyvis_network = Network(
            height=plot_height,
            width=plot_width,
            filter_menu=filter_menu,
            neighborhood_highlight=neighborhood_highlight,
            notebook=False,
        )
        # hack to hide labels: font_color="#10000000"

        pyvis_network.force_atlas_2based()

        if show_physics:
            pyvis_network.show_buttons(filter_=["physics"])
        if other_buttons:
            pyvis_network.show_buttons()

        # if initial_positions:
        #    for i, (label, position) in enumerate(zip(vertex_labels, initial_positions):
        #        x = position[:, 0]
        #        y = position[:, 1]
        #        nt.add_node(i, x=x, y=y, label=label)
        # else:
        for i, (label, frac, radius, mass) in enumerate(
            zip(vertex_labels, fractions, radii_in_plot, masses)
        ):
            pyvis_network.add_node(
                i,
                label=label,
                shape="image",
                image=_save_piechart(frac),
                size=radius,
                labelHighlightBold=labelHighlightBold,
                mass=mass,
            )

        for edge, length, width in zip(
            edges, edge_lengths_in_plot, edge_widths_in_plot
        ):
            i, j = edge
            pyvis_network.add_edge(
                i, j, length=length, width=width, color="black", smooth=edges_smooth
            )

        pyvis_network.add_node(
            -1,
            x=0,
            y=0,
            shape="image",
            image=_save_plot_legend(classes, colors),
            size=legend_scale,
            physics=False,
        )

        pyvis_network.show("visualizations/pyvis_test.html", notebook=False)
    else:
        fig = plt.figure()
        ax = fig.gca()

        plot_2d_simplicial_complex(
            simplicial_complex, normalized_radii * max_vertex_size, positions, ax=ax
        )

        if plot_letters:
            # cover_element_lable_offsets = [[-0.03, -0.03] for radius in normalized_radii]
            # cover_element_lable_offsets = [[-0.08, -0.08] for radius in normalized_radii]
            cover_element_lable_offsets = [[0 + 0.05, 0] for radius in normalized_radii]
            # draw vertex labels
            offsetted_positions = np.array(positions) + cover_element_lable_offsets
            scatter_letters(
                offsetted_positions,
                [ith_letter(i) for i in range(n_vertices)],
                ax,
            )

        # make sure everything is in the plot
        extra_margin = 0.5
        xmin_plot = np.min(positions[:, 0])
        xmax_plot = np.max(positions[:, 0])
        ymin_plot = np.min(positions[:, 1])
        ymax_plot = np.max(positions[:, 1])
        plt.xlim([xmin_plot - extra_margin, xmax_plot + extra_margin])
        plt.ylim([ymin_plot - extra_margin, ymax_plot + extra_margin])

        ax.axis("equal")
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.axis("off")

        if plot_name:
            plt.savefig(
                plot_name + ".png",
                format="png",
                bbox_inches="tight",
                dpi=150,
            )

        return fig


def plot_persistence_barcode(persistence_diagram, plot_name=None):
    fig = plt.figure(figsize=(8, 2))
    ax = fig.gca()
    gudhi.plot_persistence_barcode(persistence_diagram, axes=ax)
    ax.set_title("")
    if plot_name:
        plt.savefig(
            plot_name + ".eps",
            format="eps",
            bbox_inches="tight",
        )
    return fig

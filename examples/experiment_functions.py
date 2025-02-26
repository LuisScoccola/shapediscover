import numpy as np
import scipy as sp
import numba as nb
import gudhi
from ripser import ripser
from persim import plot_diagrams

from shapediscover import ShapeDiscover


def correct_homology_quotient(persistence_diagrams, target_betti_numers, n_bins=1000):
    """
    Given a sequence of persistence diagrams [PD_0, ..., PD_n] and Betti numbers [betti_0, ..., betti_n], find the quotient of filtration values [t] such that [dim_t(PD_i) = betti_i].
    """

    if len(persistence_diagrams) == 0:
        return 0

    target_betti_numers = np.array(target_betti_numers)

    max_homological_dimension = len(target_betti_numers)

    # compute discretized filtration values
    min_filtration_value = min(
        [
            np.min(persistence_diagram)
            for persistence_diagram in persistence_diagrams
            if len(persistence_diagram) > 0
        ]
    )
    max_filtration_value = max(
        [
            np.max(persistence_diagram[persistence_diagram < np.inf])
            for persistence_diagram in persistence_diagrams
            if len(persistence_diagram[persistence_diagram < np.inf]) > 0
        ]
    )

    filtration_values = np.linspace(
        min_filtration_value, max_filtration_value, num=n_bins, endpoint=True
    )

    sequence_of_betti_numbers = np.zeros((n_bins, max_homological_dimension))
    for dimension, pd in enumerate(persistence_diagrams):
        for interval in pd:
            start = np.searchsorted(filtration_values, interval[0])
            end = np.searchsorted(filtration_values, interval[1])
            sequence_of_betti_numbers[start:end, dimension] += 1

    tot_correct = 0
    for betti_numbers in sequence_of_betti_numbers:
        if np.array_equal(betti_numbers, target_betti_numers):
            tot_correct += 1

    return tot_correct / len(sequence_of_betti_numbers)


def binary_search(
    function_to_evaluate,
    min_parameter,
    max_parameter,
    target,
    max_iter,
    record_extra=False,
):
    left = min_parameter
    right = max_parameter
    low_bound = -np.inf
    up_bound = np.inf

    iter_n = 0
    best_extra = None
    while left < right:
        if iter_n >= max_iter:
            print("Binary search did not coverge")
            break
        iter_n += 1

        m = (left + right) // 2
        if record_extra:
            curr_val, extra = function_to_evaluate(m)
        else:
            curr_val = function_to_evaluate(m)
        if curr_val < target:
            left = m + 1
            if low_bound < curr_val:
                low_bound = curr_val
        else:
            right = m
            if record_extra:
                best_extra = extra
            if up_bound > curr_val:
                up_bound = curr_val

    if record_extra:
        return left, best_extra, low_bound, up_bound
    else:
        return left, low_bound, up_bound


def test_correct_homology_quotient_discover(
    X,
    target_betti_numbers,
    target_recovery,
    min_n_cover,
    max_n_cover,
    model="set_function",
    max_iter=20,
    knn=15,
    loss_weights=[1, 0, 0, 10],
    early_stop_tolerance=1e-4,
    verbose=True,
):

    max_homological_dimension = len(target_betti_numbers) - 1

    def test_ph_homology(n_cover):
        n_cover = int(n_cover)
        discover = ShapeDiscover(
            n_cover,
            knn=knn,
            loss_weights=loss_weights,
            early_stop_tolerance=early_stop_tolerance,
            model=model,
        )
        discover.fit(X, verbose=False, plot_loss_curve=False)
        discover.fit_persistence(max_homological_dimension, verbose=False)
        correct_quotient = correct_homology_quotient(
            discover.persistence_diagram_, target_betti_numbers
        )
        if verbose:
            print("number of covers:", str(n_cover), "quotient:", str(correct_quotient))
        return correct_quotient, discover.simplex_tree_.num_simplices()

    return binary_search(
        test_ph_homology,
        min_n_cover,
        max_n_cover,
        target_recovery,
        max_iter,
        record_extra=True,
    )


def test_correct_homology_subsample_ripser(
    X,
    target_betti_numbers,
    target_recovery,
    min_n_points,
    max_n_points,
    max_iter=20,
    verbose=True,
):
    max_homological_dimension = len(target_betti_numbers) - 1

    X_subsample_prefix = np.array(
        gudhi.subsampling.choose_n_farthest_points(
            X, nb_points=max_n_points, starting_point=0
        )
    )

    def test_ph_homology(n_subsample):
        X = X_subsample_prefix[:n_subsample]

        dgms = ripser(X, maxdim=max_homological_dimension)["dgms"]

        if verbose:
            plot_diagrams(dgms, show=True)

        correct_quotient = correct_homology_quotient(dgms, target_betti_numbers)

        if verbose:
            print("sample size:", str(n_subsample), "quotient:", str(correct_quotient))

        return correct_quotient, X

    return binary_search(
        test_ph_homology,
        min_n_points,
        max_n_points,
        target_recovery,
        max_iter,
        record_extra=True,
    )


def test_correct_homology_subsample_alpha(
    X,
    target_betti_numbers,
    target_recovery,
    min_n_points,
    max_n_points,
    max_iter=20,
    verbose=True,
):
    max_homological_dimension = len(target_betti_numbers) - 1

    X_subsample_prefix = np.array(
        gudhi.subsampling.choose_n_farthest_points(
            X, nb_points=max_n_points, starting_point=0
        )
    )

    def test_ph_homology(n_subsample):

        X = X_subsample_prefix[:n_subsample]
        simplex_tree = gudhi.AlphaComplex(points=X).create_simplex_tree()

        gudhi_pd = simplex_tree.persistence()
        if verbose:
            gudhi.plot_persistence_barcode(gudhi_pd)

        pd = [
            np.array(simplex_tree.persistence_intervals_in_dimension(i))
            for i in range(max_homological_dimension + 1)
        ]

        correct_quotient = correct_homology_quotient(pd, target_betti_numbers)


        if verbose:
            print("sample size:", str(n_subsample), "quotient:", str(correct_quotient))

        simplicial_complex_size = simplex_tree.num_simplices()

        return correct_quotient, simplicial_complex_size


    return binary_search(
        test_ph_homology,
        min_n_points,
        max_n_points,
        target_recovery,
        max_iter,
        record_extra=True,
    )



def size_vietoris_rips_edgecollapse(pointcloud, max_dim, n_iter=1, plot_pd=False):
    simplex_tree = gudhi.RipsComplex(points=pointcloud).create_simplex_tree(
        max_dimension=max_dim
    )

    initial_num_simplices = simplex_tree.num_simplices()

    if n_iter > 0:
        simplex_tree.collapse_edges(n_iter)
    else:
        n_sim = simplex_tree.num_simplices()
        simplex_tree.collapse_edges(1)
        while simplex_tree.num_simplices() < n_sim:
            n_sim = simplex_tree.num_simplices()
            simplex_tree.collapse_edges(1)

    simplex_tree.expansion(max_dim)

    if plot_pd:
        gudhi.plot_persistence_diagram(simplex_tree.persistence())

    final_num_simplices = simplex_tree.num_simplices()

    return initial_num_simplices, final_num_simplices


def pointcloud_to_perseus_rips(pointcloud, max_homological_dimension, file_name):
    n_points = pointcloud.shape[0]

    fraction_of_min_dist = 10

    all_dists = sp.spatial.distance_matrix(pointcloud, pointcloud)
    min_dist = np.min(all_dists[all_dists > 0])
    max_dist = np.max(all_dists)

    step_size = min_dist / fraction_of_min_dist
    n_steps = int(max_dist // step_size) + 1
    initial_threshold_distance = 0.0

    with open(file_name, "w+") as out_file:
        out_file.write(str(n_points) + "\n")
        out_file.write(
            str(initial_threshold_distance)
            + " "
            + str(step_size)
            + " "
            + str(n_steps)
            + " "
            + str(max_homological_dimension)
            + "\n"
        )

        for distances in all_dists:
            for i in range(n_points):
                out_file.write(str(distances[i]) + " ")
            out_file.write("0" + "\n")


def test_correct_homology_quotient_all(
    data_name,
    X,
    target_betti_numbers,
    target_recovery,
    # shapediscover parameters
    min_n_cover,
    max_n_cover,
    knn,
    loss_weights,
    # rips parameters
    min_n_points_rips,
    max_n_points_rips,
    # sparse rips parameters
    min_n_points_sparserips,
    max_n_points_sparserips,
    approx_sparserips1,
    approx_sparserips2,
    # witness parameters
    min_n_points_witness,
    max_n_points_witness,
    # alpha parameters
    min_n_points_alpha,
    max_n_points_alpha,
    to_test = [
        "shapediscover",
        "rips",
        "witness",
        "alpha"
    ],
    verbose=False,
):
    print(data_name, target_recovery)

    if "shapediscover" in to_test:
        # ShapeDiscover
        (
            best_parameter_shapediscover,
            best_size_shapediscover,
            low_bound_shapediscover,
            upper_bound_shapediscover,
        ) = test_correct_homology_quotient_discover(
            X,
            target_betti_numbers,
            target_recovery,
            min_n_cover=min_n_cover,
            max_n_cover=max_n_cover,
            knn=knn,
            loss_weights=loss_weights,
            verbose=verbose,
        )
        print("shapediscover parameter", best_parameter_shapediscover)
        print("shapediscover size", best_size_shapediscover)

        #(
        #    best_parameter_shapediscovergnn,
        #    best_size_shapediscovergnn,
        #    low_bound_shapediscovergnn,
        #    upper_bound_shapediscovergnn,
        #) = test_correct_homology_quotient_discover(
        #    X,
        #    target_betti_numbers,
        #    target_recovery,
        #    model="graph_nn",
        #    min_n_cover=min_n_cover,
        #    max_n_cover=max_n_cover,
        #    knn=knn,
        #    loss_weights=[1,0,0,10],
        #    verbose=verbose,
        #)
        #print("shapediscovergnn parameter", best_parameter_shapediscovergnn)
        #print("shapediscovergnn size", best_size_shapediscovergnn)

    if "rips" in to_test:
        # Rips
        best_parameter_rips, sample, low_bound_rips, upper_bound_rips = (
            test_correct_homology_subsample_ripser(
                X,
                target_betti_numbers,
                target_recovery,
                min_n_points=min_n_points_rips,
                max_n_points=max_n_points_rips,
                verbose=verbose,
            )
        )
        print("rips parameter", best_parameter_rips)
        print("rips bounds quotient", low_bound_rips, upper_bound_rips)

        # Rips edge collapse
        best_size_rips, best_size_rips_edgecollapse = size_vietoris_rips_edgecollapse(
            sample, len(target_betti_numbers)
        )
        print("rips size", best_size_rips)
        print("rips edgecollapse", best_size_rips_edgecollapse)

        # Sparse Rips 1
        (
            best_parameter_sparserips1,
            best_size_sparserips1,
            low_bound_sparserips1,
            upper_bound_sparserips1,
        ) = test_correct_homology_subsample_sparserips(
            X,
            target_betti_numbers,
            target_recovery,
            min_n_points_sparserips,
            max_n_points_sparserips,
            approx_sparserips1,
            verbose=verbose,
        )
        print("sparserips1 parameter", best_parameter_sparserips1)
        print("sparserips1 size", best_size_sparserips1)
        print("sparserips1 bounds quotient", low_bound_sparserips1, upper_bound_sparserips1)

        # Sparse Rips 2
        (
            best_parameter_sparserips2,
            best_size_sparserips2,
            low_bound_sparserips2,
            upper_bound_sparserips2,
        ) = test_correct_homology_subsample_sparserips(
            X,
            target_betti_numbers,
            target_recovery,
            min_n_points_sparserips,
            max_n_points_sparserips,
            approx_sparserips2,
            verbose=verbose,
        )
        print("sparserips2 parameter", best_parameter_sparserips2)
        print("sparserips2 size", best_size_sparserips2)
        print("sparserips2 bounds quotient", low_bound_sparserips2, upper_bound_sparserips2)

        # save pointcloud for perseus
        pointcloud_to_perseus_rips(
            sample,
            len(target_betti_numbers),
            "external_code/perseus/perseus_data/" + data_name + str(sample.shape[0]),
        )

    if "witness" in to_test:
        # witness 0
        best_parameter_witness0, best_size_witness0, low_bound_witness0, upper_bound_witness0 = test_correct_homology_witness(
            X,
            target_betti_numbers,
            target_recovery,
            min_n_points_witness,
            max_n_points_witness,
            nu=0,
            verbose=verbose,
        )

        print("witness0 parameter", best_parameter_witness0)
        print("witness0 size", best_size_witness0)
        print("bounds witness0", low_bound_witness0, upper_bound_witness0)

        # witness 1
        best_parameter_witness1, best_size_witness1, low_bound_witness1, upper_bound_witness1 = test_correct_homology_witness(
            X,
            target_betti_numbers,
            target_recovery,
            min_n_points_witness,
            max_n_points_witness,
            nu=1,
            verbose=verbose,
        )

        print("witness1 parameter", best_parameter_witness1)
        print("witness1 size", best_size_witness1)
        print("bounds witness1", low_bound_witness1, upper_bound_witness1)

        # witness 2
        best_parameter_witness2, best_size_witness2, low_bound_witness2, upper_bound_witness2 = test_correct_homology_witness(
            X,
            target_betti_numbers,
            target_recovery,
            min_n_points_witness,
            max_n_points_witness,
            nu=2,
            verbose=verbose,
        )

        print("witness2 parameter", best_parameter_witness2)
        print("witness2 size", best_size_witness2)
        print("bounds witness2", low_bound_witness2, upper_bound_witness2)

    if "alpha" in to_test:
        best_parameter_alpha, best_size_alpha, low_bound_alpha, upper_bound_alpha= (
            test_correct_homology_subsample_alpha(
                X,
                target_betti_numbers,
                target_recovery,
                min_n_points=min_n_points_alpha,
                max_n_points=max_n_points_alpha,
                verbose=verbose,
            )
        )
        print("alpha parameter", best_parameter_alpha)
        print("alpha size", best_size_alpha)
        print("alpha bounds quotient", low_bound_alpha, upper_bound_alpha)

def get_correct_homology_quotient_all(
    data_name,
    X,
    target_betti_numbers,
    n_vertices,
    # shapediscover parameters
    knn,
    loss_weights,
    to_test = [
        "shapediscover",
        "rips",
        "witness",
        "alpha"
    ],
):

    print(data_name, n_vertices)

    max_homological_dimension = len(target_betti_numbers) - 1

    if "shapediscover" in to_test:
        n_cover = int(n_vertices)

        discover = ShapeDiscover(
            n_cover,
            knn=knn,
            loss_weights=loss_weights,
        )
        discover.fit(X, verbose=True, plot_loss_curve=False)

        discover.fit_persistence(max_homological_dimension, verbose=False)
        correct_quotient_shapediscover = correct_homology_quotient(
            discover.persistence_diagram_, target_betti_numbers
        )

        print("shapediscover", correct_quotient_shapediscover)


    X_subsample = np.array(
        gudhi.subsampling.choose_n_farthest_points(
            X, nb_points=n_vertices, starting_point=0
        )
    )

    if "rips" in to_test:
        dgms = ripser(X_subsample, maxdim=max_homological_dimension)["dgms"]
        correct_quotient_rips = correct_homology_quotient(dgms, target_betti_numbers)
        print("rips", correct_quotient_rips)


    if "witness" in to_test:
        for nu in [0,1,2]:
            simplex_tree = witness_complex(
                X, X_subsample, max_homological_dimension + 1, nu=nu
            )

            gudhi_pd = simplex_tree.persistence()

            pd = [
                np.array(simplex_tree.persistence_intervals_in_dimension(i))
                for i in range(max_homological_dimension + 1)
            ]

            correct_quotient_witness = correct_homology_quotient(pd, target_betti_numbers)

            print("witness", nu, correct_quotient_witness)

    if "alpha" in to_test:

        simplex_tree = gudhi.AlphaComplex(points=X_subsample).create_simplex_tree()

        gudhi_pd = simplex_tree.persistence()

        pd = [
            np.array(simplex_tree.persistence_intervals_in_dimension(i))
            for i in range(max_homological_dimension + 1)
        ]

        correct_quotient_alpha = correct_homology_quotient(pd, target_betti_numbers)


        print("alpha", correct_quotient_alpha)


def test_correct_homology_subsample_sparserips(
    X,
    target_betti_numbers,
    target_recovery,
    min_n_points,
    max_n_points,
    approx,
    max_iter=20,
    verbose=True,
):
    max_homological_dimension = len(target_betti_numbers) - 1

    X_subsample_prefix = np.array(
        gudhi.subsampling.choose_n_farthest_points(
            X, nb_points=max_n_points, starting_point=0
        )
    )

    def test_ph_homology(n_subsample):
        X = X_subsample_prefix[:n_subsample]
        simplex_tree = gudhi.RipsComplex(points=X, sparse=approx).create_simplex_tree(
            max_dimension=max_homological_dimension + 1
        )

        gudhi_pd = simplex_tree.persistence()
        if verbose:
            gudhi.plot_persistence_barcode(gudhi_pd)

        pd = [
            np.array(simplex_tree.persistence_intervals_in_dimension(i))
            for i in range(max_homological_dimension + 1)
        ]

        correct_quotient = correct_homology_quotient(pd, target_betti_numbers)

        if verbose:
            print("sample size:", str(n_subsample), "quotient:", str(correct_quotient))

        simplicial_complex_size = simplex_tree.num_simplices()

        return correct_quotient, simplicial_complex_size

    return binary_search(
        test_ph_homology,
        min_n_points,
        max_n_points,
        target_recovery,
        max_iter,
        record_extra=True,
    )


def test_correct_homology_witness(
    X,
    target_betti_numbers,
    target_recovery,
    min_n_points,
    max_n_points,
    nu=0,
    max_iter=20,
    verbose=True,
):
    max_homological_dimension = len(target_betti_numbers) - 1

    X_subsample_prefix = np.array(
        gudhi.subsampling.choose_n_farthest_points(
            X, nb_points=max_n_points, starting_point=0
        )
    )

    def test_ph_homology(n_subsample):
        X_subsample = X_subsample_prefix[:n_subsample]
        simplex_tree = witness_complex(
            X, X_subsample, max_homological_dimension + 1, nu=nu
        )

        gudhi_pd = simplex_tree.persistence()
        if verbose:
            gudhi.plot_persistence_barcode(gudhi_pd)

        pd = [
            np.array(simplex_tree.persistence_intervals_in_dimension(i))
            for i in range(max_homological_dimension + 1)
        ]

        correct_quotient = correct_homology_quotient(pd, target_betti_numbers)

        if verbose:
            print("sample size:", str(n_subsample), "quotient:", str(correct_quotient))

        simplicial_complex_size = simplex_tree.num_simplices()

        return correct_quotient, simplicial_complex_size

    return binary_search(
        test_ph_homology,
        min_n_points,
        max_n_points,
        target_recovery,
        max_iter,
        record_extra=True,
    )


def witness_complex(pointcloud, landmarks, max_dimension, nu=0):
    n_vertices = landmarks.shape[0]
    n_points = pointcloud.shape[0]
    n_edges = (n_vertices * (n_vertices - 1)) // 2
    edges = np.zeros((n_edges, 2), dtype=int)
    edges_births = np.full(n_edges, np.inf)

    if nu == 0:
        relaxation = np.zeros(n_points)
    else:
        all_dists = sp.spatial.distance_matrix(pointcloud, landmarks)
        relaxation = np.partition(all_dists, nu, axis=1)[:, nu]

    @nb.jit(nopython=True)
    def _witness_complex_main_loop(
        pointcloud, landmarks, n_vertices, n_points, edges, edges_births
    ):
        k = 0
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                edges[k, 0], edges[k, 1] = i, j
                for x_index in range(n_points):
                    birth_according_to_x = max(
                        max(
                            np.linalg.norm(landmarks[i] - pointcloud[x_index]),
                            np.linalg.norm(landmarks[j] - pointcloud[x_index]),
                        )
                        - relaxation[x_index],
                        0,
                    )
                    if birth_according_to_x < edges_births[k]:
                        edges_births[k] = birth_according_to_x
                k += 1

    _witness_complex_main_loop(
        pointcloud, landmarks, n_vertices, n_points, edges, edges_births
    )

    simplex_tree = gudhi.SimplexTree()
    vertices = np.arange(n_vertices).reshape(-1, 1)
    simplex_tree.insert_batch(vertices.T, np.zeros(n_vertices))
    simplex_tree.insert_batch(edges.T, edges_births)
    simplex_tree.expansion(max_dimension)

    return simplex_tree

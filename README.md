# ShapeDiscover

Learn covers of data with geometric and topological optimization,
for topological inference and visualization.
See [[SLH, ICML25]](#1) for background on _cover learning_ and _topological inference_.


> [!Note]
> Alpha version. User-facing interface is subject to breaking changes.

## Installation

Basic installation:

```pip install .```

Some examples require extra libraries that can be installed with:

```pip install ".[extras]"```

## Examples

Here are two small example using `ShapeDiscoverLite`, which is the currently recommended interface.
See notebooks in the `examples` directory for more examples.

### Topological inference
Recovering the topology of a two-dimensional sphere.
We choose a cover with 25 elements for illustration purposes, but ShapeDiscover recovers the correct topology with as few as 5 cover elements.

```python
from shapediscover import ShapeDiscoverLite, FuzzyCoverPersistence
import gudhi
from synthetic_data import sphere

X = sphere(2000, 2)
coverer = ShapeDiscoverLite(25)
fuzzy_cover = coverer.fit_transform(X)

persistence_barcode = FuzzyCoverPersistence(max_dimension=2, log_rescaling=True).fit_transform(fuzzy_cover)
gudhi.plot_persistence_barcode(persistence_barcode)
plt.show()
```

![Alt text](https://github.com/LuisScoccola/shapediscover/blob/main/docs/figures/sphere_barcode.png)

### Visualization

We visualize the MNIST handwritten digits dataset.

```python
from shapediscover import plot_nerve
import torchvision

mnist_dataset = torchvision.datasets.MNIST(root="./datasets", download=True)
X = np.array([np.array(image_label[0]).flatten() for image_label in mnist_dataset])
y = np.array([image_label[1] for image_label in mnist_dataset])

coverer = ShapeDiscoverLite(10,regularization=40)
fuzzy_cover = coverer.fit_transform(X)
plot_nerve(fuzzy_cover, threshold=0.8, interactive=True, max_vertex_size=0.8, labels=y)
```

![Alt text](https://github.com/LuisScoccola/shapediscover/blob/main/docs/figures/MNIST_nerve.png)

The output of ShapeDiscover on the left, and UMAP's two-dimensional projection on the right, for comparison.


## Authors

[Luis Scoccola](https://luisscoccola.com/) and [Uzu Lim](https://sites.google.com/view/uzulim/main).

## References

<a id="1">[SLH, ICML25]</a> 
*Cover learning for large-scale topology representation*. Luis Scoccola, Uzu Lim, Heather A. Harrington. International Conference on Machine Learning (ICML 2025)

## License

This software is published under the 3-clause BSD license.

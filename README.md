# ShapeDiscover

Learn covers of geometric data with geometric and topological optimization, using the methodology of [[1]](#1).

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

```python
from shapediscover import ShapeDiscoverLite, FuzzyCoverPersistence, plot_nerve
import gudhi as gudhi
from synthetic_data import sphere

X = sphere(2000, 2)
coverer = ShapeDiscoverLite(25)
fuzzy_cover = coverer.fit_transform(X)

persistence_diagram = FuzzyCoverPersistence(max_dimension=2, log_rescaling=True).fit_transform(fuzzy_cover)
gudhi.plot_persistence_barcode(persistence_diagram)
plt.show()
```




## Authors

[Luis Scoccola](https://luisscoccola.com/) and [Uzu Lim](https://sites.google.com/view/uzulim/main).

## References

<a id="1">[1]</a> 
*Cover learning for large-scale topology representation*. Luis Scoccola, Uzu Lim, Heather A. Harrington. International Conference on Machine Learning (ICML 2025)

## License

This software is published under the 3-clause BSD license.

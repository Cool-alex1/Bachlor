import itertools

import numpy as np
import pytest
import xarray as xr
import matplotlib.pyplot as plt
import marigold as mg 
import numpy as np
from knano.cluster import Cluster
import knano.transform as transform
from tqdm.auto import tqdm, trange

# dataset = np.ones((5, 10), dtype=np.double)
# dataset[2:4, 3:7] = 0
# print(dataset)

# result = mg.marigold(X=dataset, n_clusters=2, init="first")

# print(result)


data_shape = (20, 20, 30, 10)
arr = np.random.randn(*data_shape)
dimension_labels = ["x", "y", "e", "k"]
coords = {d: np.arange(s) for d, s in zip(dimension_labels, arr.shape)}
xarr = xr.DataArray(data=arr, dims=dimension_labels, coords=coords, name="raw")
xdset = xarr.to_dataset()

cl = Cluster(xdset)
cl.navigation_dimensions = ['x','y']
cl.signal_dimensions = ['e','k']

fig,ax = plt.subplots(1,2)
cl.source.mean(cl.navigation_dimensions).plot(ax=ax[0])
cl.source.sum(cl.signal_dimensions).plot(ax=ax[1])
cl.pre_process([ # a list of transformations we want to apply to each spectrum before feeding it to kmeans
        transform.Resize((128,128)), # reshape the image
        transform.Rotate(90), # rotate by some angle
        transform.ToTensor(), # necessary for DCT
        transform.DCT_2D(n_components=32), # compute the dct transform and pick the first n components
        transform.ToNumpy(), # transform back to np.ndarrays (scikit kmeans needs this as input)
        transform.Flatten(), # kmeans works on 1D vectors, not images
    ], )
cl.fit(n_clusters=10,)

plt.figure()
plt.imshow(cl.cluster_map)

plt.show()

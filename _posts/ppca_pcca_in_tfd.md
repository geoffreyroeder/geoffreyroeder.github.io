# Imports and setup



```python
from __future__ import annotations
```

```python
import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import types
from collections import namedtuple
from itertools import chain 
from functools import partial
import scipy
from jax.experimental import optimizers
from scipy.stats import ortho_group
from sklearn.cross_decomposition import CCA

plt.style.use("ggplot")
warnings.filterwarnings('ignore')
```

import plotly.graph_objects as go   # for visualizable plots

```python
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
from jax.ops import index_update, index
from jax import grad
from jax import jit
from jax import random
from jax import value_and_grad
from jax import vmap
```

```python
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels
```

```python
import networkx as nx
import pydot as dot
from IPython.core.display import SVG
```

## Plot PGM

```python
# process data into adjacency list
# TODO: nicer, d3.js or something?
def draw_pgm(joint_dist, special_nodes=None):
    graph = dot.Dot(graph_type='digraph')
    for node, parents in joint_dist.resolve_graph():
      pnode = dot.Node(node, shape='circle')
      graph.add_node(pnode)
      for parent in parents:
#         print(shape)
#         print(parent)
        par_pnode = dot.Node(parent, shape='circle')
        graph.add_node(par_pnode)
        graph.add_edge(dot.Edge(par_pnode, pnode))

    img = graph.create_svg(prog='dot')
    return SVG(img)
```

y[2*latent_dim:]# Define Probabilistic PCA model
We first implement PPCA using TensorFlow Distributions with a JAX backend. This forms the "decorrelator" component of a MERA ansatz.


```python
A = jnp.array(range(16)).reshape([4,4])
A[...,0]  # ... syntax means "however many", in case you don't know beforehand
```

    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)





    DeviceArray([ 0,  4,  8, 12], dtype=int32)



```python
# Root of a JointDistributionCoroutine is a distribution with no parents
Root = tfd.JointDistributionCoroutine.Root 


def build_PPCA_model(data_dim, latent_dim, num_datapoints, stddv_datapoints):
  """Create a Model object for PPCA specialized to a data and latent dimension

  Args:
    data_dim: integer number corresponding to observed data dimension
    latent_dim: integer number corresponding to latent variable dimension
    num_datapoints: integer number of data samples 
    stddv_datapoints: standard deviation corresponding to noise in observed data
  
  Returns:
    Model representing Probabilistic Principal Components Analysis 
  """
  W = yield Root(tfd.Normal(loc=jnp.zeros([data_dim, latent_dim]),
                 scale=4.0*jnp.ones([data_dim, latent_dim]),
                 name="W"))
  mu = yield Root(tfd.Normal(loc=3*jnp.ones([data_dim, 1]),
                 scale=.5*jnp.ones([data_dim, 1]), 
                 name="mu"))
  z = yield Root(tfd.Normal(loc=jnp.zeros([latent_dim, num_datapoints]),
                 scale=.5*jnp.ones([latent_dim, num_datapoints]),
                 name="z"))
  x = yield tfd.Normal(loc=jnp.matmul(W, z) + mu,
                       scale=stddv_datapoints,
                       name="x")
```

```python
mu = lambda: tfd.Normal(loc=3*jnp.ones([data_dim, 1]),
                 scale=.5*jnp.ones([data_dim, 1]), 
                 name="mu")
```

Acceptance test decorrelator (PPCA)

```python
num_datapoints = 1000
data_dim = 2
latent_dim = 2
stddv_datapoints = 2 # sigma

PPCA_model = functools.partial(build_PPCA_model,
    data_dim=data_dim,
    latent_dim=latent_dim,
    num_datapoints=num_datapoints,
    stddv_datapoints=stddv_datapoints)

ppca_joint_dist = tfd.JointDistributionCoroutine(PPCA_model)
```

## Visualization: model is PCA where W contains principal axes (not necessarily orthogonal)

```python
def principal_axes(X):
  """Find principal axes of a 2D square matrix X

  Args:
    X: 2-D jax.numpy.ndarry

  Returns:
    Matrix of right singular vectors of X, e.g., V, where X = UDV^T
  """
  # X is assumed to be data_dim x num_datapoints
  # subtract column means to center
  X_c = (X - jnp.mean(X, axis=1, keepdims=True)).T
  U, D, V_T = jnp.linalg.svd(X_c.T @ X_c)
  return V_T.T  # Columns of V contain principal directions
```

## Verify: plot true and sampled principal axes from random PPCA model

```python
def plot_ppca(W, mu, z, X):
    sns.set_style("darkgrid", {"axes.facecolor": ".87"})

    W_true = principal_axes(jnp.dot(W, z))
    W_ex = scipy.linalg.orth(W)

    # Center of sampled data
    data_centroid = jnp.mean(X, axis=1)


    arrows = []  # capture object refernces to label
    points = []

    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)


    # Plot samples z
    points.append(plt.scatter(z[0, :], z[1, :],  marker="*", color='red', alpha=0.4))

    # Plot sampled data points x
    points.append(ax.scatter(X[0, :], X[1, :], marker="v", color='blue', alpha=0.4))

    # Plot samples z transformed through W 
    Wz_mu = jnp.matmul(W, z) + mu
    points.append(ax.scatter(Wz_mu[0, :], Wz_mu[1, :], marker="^", color='limegreen', alpha=0.4))

    arrows = []  # capture object refernces to label


    # plot orthogonalized basis directions for true generative W basis 
    for p_ax in W_ex:
      arrows.append(ax.axline([0,0] , p_ax, ls="--", c="r", alpha=0.5))

    # plot sample covariance matrix principal axes
    for p_ax in principal_axes(X):
       #, ls="--", c="r", alpha=0.5)
      # print(data_centroid) [x, y]
      arrows.append(ax.axline(data_centroid, p_ax+data_centroid, ls="-", c="blue", alpha=0.5))
      # arrows.append(ax.arrow(*data_centroid , *14*p_ax, width=.15, head_length=.25, 
      #                         fc='blue', ec='blue', zorder=1))

    # plot true generative model principal axes
    for p_ax in principal_axes(Wz_mu):
      arrows.append(ax.axline(jnp.squeeze(mu), p_ax+jnp.squeeze(mu), ls="-.", 
                              c="lime", alpha=1))
      # arrows.append(ax.arrow(*jnp.squeeze(mu) , *12*p_ax, width=.10, head_length=.25, 
      #                         fc='limegreen', ec='limegreen', zorder=1))

    to_label = chain(*zip(points,arrows[::2]))  #interleave points and arrows
    labels = ["$z \sim \mathcal{N}(0, I)$",
              "$W$ (orthoganalized) principal axes",
              "$X \sim \mathcal{N}(Wz + \mu, \sigma^2I)$", 
                       "X principle axes",
              "Wz+$\mu$",

              "Wz+$\mu$ principal axes"]

    ax.legend(to_label, labels)

    ax.set_aspect("equal", "datalim")
    ax.set_title("PPCA: True Generative Process vs. Noisy Sampled Principal Axes")
    fig.show()
```

```python
key = random.PRNGKey(42)
new_keys, subkeys = random.split(key)
actual_W, actual_mu, actual_z, X_train = ppca_joint_dist.sample(seed=key)
plot_ppca(actual_W, actual_mu, actual_z, X_train)

```


![png](ppca_pcca_in_tfd_files/output_18_0.png)


# Define Probabilistic CCA model


```python
# Root of a JointDistributionCoroutine is a distribution with no parents
Root = tfd.JointDistributionCoroutine.Root 


def build_PCCA_model(data_dim1, data_dim2, latent_dim, num_datapoints, std_x1, 
                     std_x2):
  """Create a Model for Probabilistic Canonical Correlations Analysis
  Args:
    data_dim: integer number corresponding to observed data dimension
    latent_dim: integer number corresponding to latent variable dimension
    num_datapoints: integer number of data samples 
    stddv_datapoints: standard deviation corresponding to noise in observed data
  
  Returns:
    Model representing Probabilistic Principal Components Analysis 
  """
  z = yield Root(tfd.Normal(loc=jnp.zeros([latent_dim, num_datapoints]),
                scale=.1*jnp.ones([latent_dim, num_datapoints]),
                name="z"))
  W1 = yield Root(tfd.Normal(loc=jnp.zeros([data_dim1, latent_dim]),
                 scale=4.0*jnp.ones([data_dim1, latent_dim]),
                 name="W1"))
  W2 = yield Root(tfd.Normal(loc=jnp.zeros([data_dim2, latent_dim]),
                scale=2.0*jnp.ones([data_dim2, latent_dim]),
                name="W2"))
  mu1 = yield Root(tfd.Normal(loc=3*jnp.ones([data_dim1, 1]),
                 scale=10*jnp.ones([data_dim1, 1]), 
                 name="mu1"))
  mu2 = yield Root(tfd.Normal(loc=3*jnp.ones([data_dim2, 1]),
                scale=-5*jnp.ones([data_dim2, 1]), 
                name="mu2"))
  x1 = yield tfd.Normal(loc=jnp.matmul(W1, z) + mu1,
                       scale=std_x1,
                       name="x1")
  x2 = yield tfd.Normal(loc=jnp.matmul(W2, z) + mu2,
                      scale=std_x2,
                      name="x2")
```

## Sample from joint PCCA: 2D case


```python
num_datapoints = 2000 
data_dim1 = 2
data_dim2 = 2
latent_dim = 2
std_x1 = .005
std_x2 = .5

key = random.PRNGKey(42)
new_keys, subkeys = random.split(key)

# N1 = random.normal(key, shape=[data_dim1, data_dim1])
# rand_cov1 = jnp.matmul(N1.T, N1)
# N2 = random.normal(key, shape=[data_dim2, data_dim2])
# rand_cov2 = jnp.matmul(N2.T, N2)

# specify the values of the hyperparameters
# num_datapoints will be the batch size in any gradient-based learning algorithm
# e.g., X is a batch of data coming in
# functools.partial(build_model, num_datapoints=X.shape[0])
PCCA_model = functools.partial(build_PCCA_model,
    data_dim1=data_dim1,
    data_dim2=data_dim2,
    latent_dim=latent_dim,
    num_datapoints=num_datapoints,
    std_x1=std_x1,
    std_x2=std_x2)

joint_dist_pcca = tfd.JointDistributionCoroutine(PCCA_model)
# draw_pgm(joint_dist_pcca.)
```

```python
key = random.PRNGKey(42)
new_keys, subkeys = random.split(key)
actual_z, actual_W1, actual_W2, actual_mu1, actual_mu2, X1_train, X2_train = joint_dist_pcca.sample(seed=key)
print(actual_z)
```

    [[-0.01070893  0.1234409  -0.06353103 ... -0.03674607 -0.19175364
      -0.10487504]
     [-0.12384464  0.18903953 -0.02106112 ... -0.2081848   0.14156266
       0.1187698 ]]


```python
# def plot_paxes(W, offset, arrows, ax, ls="--", c="r", alpha=0.5):
#   for col_vec in W:
    

origin = jnp.array([[0,0]])

W1z = jnp.matmul(actual_W1, actual_z) 
W2z = jnp.matmul(actual_W2, actual_z)  # found an error here

W1z_mu1 = jnp.matmul(actual_W1, actual_z) + actual_mu1
W2z_mu2 = jnp.matmul(actual_W2, actual_z) + actual_mu2

to_plot = [actual_W1, 
          actual_W2, 
          X1_train[:2], 
          X2_train[:2], 
          W1z_mu1, 
          W2z_mu2]
centroids = [origin, origin, jnp.mean(X1_train[:2], axis=1, keepdims=True),
             jnp.mean(X2_train[:2], axis=1, keepdims=True), actual_mu1, 
             actual_mu2]
cols = ['blue', 'green', 'blue', 'green', 'lightblue', 'lightgreen', 'cyan', 'lime']
lts = [':', ':', '-', '-','-.', '-.', ':',':']             
double = lambda lst: list(chain(*zip(lst,lst)))
paxes =[principal_axes(y) for y in to_plot]
```

```python
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
sns.set_style("darkgrid", {"axes.facecolor": ".87"})

points = []

# Plot noise samples z
points.append(plt.scatter(actual_z[0, :], actual_z[1, :],  marker="*", 
                          color='red', alpha=0.4))

# Plot sampled data points x1 and x2
points.append(ax.scatter(X1_train[0, :], X1_train[1, :], marker="v", 
                          color='blue', alpha=0.4))
points.append(ax.scatter(X2_train[0, :], X2_train[1, :], marker="^", 
                         color='green', alpha=0.4))

# Plot samples z transformed through W1 and W2 
points.append(ax.scatter(W1z_mu1[0, :], W1z_mu1[1, :], marker="v", 
                          color='lightblue', alpha=0.4))
points.append(ax.scatter(W2z_mu2[0, :], W2z_mu2[1, :], marker="v", 
                         color='lightgreen', alpha=0.4))

arrows = []  # capture object refernces to label

# Plot principal axes
for W, cent, col, lt in zip(paxes, centroids, cols, lts):
  # print(W)
  # print(cent)
  # print(col)
  for pax in W:
#     print(col)
#     print(pax) 
#     print(cent)
    arrows.append(ax.axline(jnp.squeeze(cent[:2]), pax[:2]+jnp.squeeze(cent[:2]), ls=lt,
                            c=col, alpha=1))

# to_label = chain(*zip(points,arrows[::2]))  #interleave points and arrows
# labels = ["$z \sim \mathcal{N}(0, I)$",
#           "$W$ (orthoganalized) principal axes",
#           "$X \sim \mathcal{N}(Wz + \mu, \sigma^2I)$", 
#                    "X principle axes",
#           "Wz+$\mu$",
 
#           "Wz+$\mu$ principal axes"]
labels = [""]
# ax.legend(to_label, labels)

ax.set_aspect("equal", "datalim")
ax.set_title("PCCA: True Generative Process vs. Noisy Sampled Canonical Components")
```




    Text(0.5, 1.0, 'PCCA: True Generative Process vs. Noisy Sampled Canonical Components')




![png](ppca_pcca_in_tfd_files/output_25_1.png)


import math
import numpy as np
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.utils import approx_min_max
from einops import rearrange


def visualize_absolute_posenc(pos_embed, plot_dir):
    """
    pos_embed should have shape [T, D]

    Create plots of absolute positional embedding.
    """
    # Plot positional encoding
    fig, ax = plt.subplots(1, 1, figsize=(0.05*pos_embed.shape[1]+2, 0.05*pos_embed.shape[0]), layout="constrained")
    im = ax.imshow(pos_embed.detach().cpu().numpy())  # pos_embed: [T, D]
    ax.set_xlabel("Embedding index")
    ax.set_ylabel("Timestep")
    fig.colorbar(im, ax=ax, shrink=0.4)
    fig.suptitle("Absolute positional embeddings")
    plt.savefig(os.path.join(plot_dir, 'pos_encoding.png'))
    plt.close()

    # Compute pairwise distance between each pair of positions
    pairwise_distances = torch.cdist(pos_embed.unsqueeze(0), pos_embed.unsqueeze(0)).squeeze(0)
    fig, ax = plt.subplots(1, 1, figsize=(0.05*pairwise_distances.shape[1], 0.05*pairwise_distances.shape[0]), layout="constrained")
    im = ax.imshow(pairwise_distances.detach().cpu().numpy())
    fig.colorbar(im)
    fig.suptitle("Pairwise distances between absolute pos encodings")
    plt.savefig(os.path.join(plot_dir, 'pos_encoding_distances.png'))
    plt.close()


def visualize_pooling_attn(pooling_attn, plot_dir):
    """
    pooling_attn assumed to have shape [B, H, T]

    For a few examples, visualize pooling attention. Each plot is an example.
    """
    n_rows = 1
    n_cols = 5
    fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows), layout="constrained")
    timesteps = np.arange(pooling_attn.shape[2])
    for example_idx in range(n_cols):
        for head_idx in range(pooling_attn.shape[1]):
            axeslist[example_idx].plot(timesteps, pooling_attn[example_idx, head_idx, :].cpu().detach().numpy(),
                                        label=f"Head {head_idx}")
        axeslist[example_idx].set_xlabel("Timestep")
        axeslist[example_idx].set_ylabel("Attn weight")
        axeslist[example_idx].set_title(f"Example {example_idx}: Pooling attn")
        axeslist[example_idx].legend()
    fig.suptitle("Pooling attention")
    plt.savefig(os.path.join(plot_dir, 'pooling_attn_weights.png'))
    plt.close()

    # pooling_attn is [B, H, T]
    min_value, max_value = approx_min_max(pooling_attn)  #  pooling_attn is [B, H, T]
    fig, axeslist = plt.subplots(n_rows, n_cols, figsize=(0.15*n_cols*pooling_attn.shape[1]+3, 0.03*n_rows*pooling_attn.shape[2]), layout="constrained")
    for c in range(n_cols):
        im = axeslist[c].imshow(rearrange(pooling_attn[c, :, :], 'h t -> t h').detach().cpu().numpy(), vmin=min_value, vmax=max_value, aspect=0.2, interpolation='none')  # stretch each column horizontally 5x)
        axeslist[c].set_xlabel("Head number")
        axeslist[c].set_ylabel("Timestep")
        axeslist[c].set_title(f"Example {c}")
    fig.colorbar(im, ax=axeslist, shrink=0.4)
    fig.suptitle("Pooling attention")
    plt.savefig(os.path.join(plot_dir, 'pooling_attn_weights_colorbar.png'))
    plt.close()


def plot_single_scatter_file(x, y, x_label, y_label, plot_dir, title_description="", filename_description="", should_align=False):
    if title_description != "":
        title_description += " - "
    if filename_description != "":
        filename_description += "_"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    title = title_description + "\n" + y_label + " vs " + x_label
    plt.figure(figsize=(8, 8))
    plot_single_scatter(x, y, x_label, y_label, title, ax=plt.gca(), should_align=should_align)
    plt.savefig(os.path.join(plot_dir, filename_description + y_label + "_vs_" + x_label + ".png"))
    plt.close()


def plot_single_scatter(x, y, x_label, y_label, title, ax=None, should_align=True):
    """Helper function which creates a single scatter plot between 2 columns, and plots to the given "ax".

    Args:
        results_df: a Pandas DataFrame containing columns with names "true_col" and "predicted_col"
        x_col: column for the x-axis
        y_col: column for the y-axis
        title: Title for the scatter-plot
        ax: The axes to plot to
        should_align: whether x and y are supposed to be scaled identically (e.g. true/predicted)
    """
    if ax is None:
        ax = plt.gca()

    # Remove rows where we don't have true OR predicted label
    not_nan = ~np.isnan(x) & ~np.isnan(y)
    y = y[not_nan]
    x = x[not_nan]
    if y.size < 2:
        return

    # Fit linear regression
    x = x.reshape(-1, 1)
    regression = LinearRegression(fit_intercept=True).fit(x, y)
    slope = regression.coef_[0]
    intercept = regression.intercept_
    regression_line = slope * x + intercept
    regression_equation = 'y={:.2f}x+{:.2f}'.format(slope, intercept)
    identity_line = x

    # Compute statistics
    y = y.ravel()
    x = x.ravel()
    r2 = r2_score(x, y)
    corr = np.corrcoef(x, y)[0, 1]
    mae = mean_absolute_error(x, y)
    mape = np.mean(100*np.abs((x - y) / (x + 1e-5)))
    rmse =  math.sqrt(mean_squared_error(x, y))

    # Plot scatterplot for this crop type
    if x.size > 500:
        density_scatter(x, y, ax=ax, s=5) #, s=2, color='black')
    else:
        ax.scatter(x, y, color="k", s=9)
    if should_align:
        stats_string = '(R^2={:.3f}, Corr={:.3f}, RMSE={:.3f})'.format(r2, corr, rmse)
    else:
        stats_string = '(Corr={:.3f})'.format(corr)
    ax.plot(x, regression_line, 'r', label=regression_equation + ' ' + stats_string) # ' (R^2={:.2f}, Corr={:.2f}, MAPE={:.2f})'.format(r2, corr, mape))
    if should_align:
        ax.plot(x, identity_line, 'g--', label='Identity function')
    ax.tick_params(labelsize=13)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title + " (num datapoints: " + str(len(x)) + ")", fontsize=13)
    ax.legend(fontsize=13)


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """Plot a scatter between x/y with density coloring (with 2d histogram).

    Code from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """

    if ax is None :
        fig, ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    norm = Normalize(vmin = np.min(z), vmax = np.max(z))

    return ax


def plot_time_series(x, filename):
    """ x should be shape [time, num_vars], numpy array
    """
    num_timesteps, num_vars = x.shape
    fig, axeslist = plt.subplots(num_vars, 1, figsize=(10, 1.5*num_vars))
    for var_idx in range(num_vars):
        axeslist[var_idx].plot(np.arange(num_timesteps), x[:, var_idx])
        axeslist[var_idx].set_xlabel('Timestep')
        axeslist[var_idx].set_ylabel('Value')
        axeslist[var_idx].set_title(f'Variable {var_idx}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

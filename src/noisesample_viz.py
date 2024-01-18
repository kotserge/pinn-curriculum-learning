import time
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from experiment.convection_curriculum_learning import ConvectionEquationPDEDataset


def plot_dataset(
    dataset: ConvectionEquationPDEDataset, figure: Figure, axes: Axes, params: dict = {}
):
    """Plots the dataset on the given figure and axes.

    Args:
        dataset (ConvectionEquationPDEDataset): The dataset.
        figure (Figure): The figure.
        axes (Axes): The axes.
        params (dict, optional): Parameters for the plot. Defaults to {}.

    Returns:
        fig, axes: The figure and axes of the plot.
    """
    labels = dataset.Y
    labels = np.reshape(labels, (params["data"]["grid"], params["data"]["grid"]))

    axes[params["plot"]["idx_grid"]][params["plot"]["idx_noise"]].imshow(
        labels,
        cmap="jet",
        aspect="auto",
        extent=params["data"]["extent"],
        origin="lower",
    )
    axes[params["plot"]["idx_grid"]][params["plot"]["idx_noise"]].set_title(
        params["title"]
    )
    axes[params["plot"]["idx_grid"]][params["plot"]["idx_noise"]].set_xlabel("x")
    axes[params["plot"]["idx_grid"]][params["plot"]["idx_noise"]].set_ylabel("t")

    return figure, axes


if __name__ == "__main__":
    """Examples for the ConvectionEquationPDEDataset."""
    grids = [8, 10, 16, 22, 32, 64]
    noises = [0.1, 1, 5, 10, 20, 30, 40, 0]

    fig, axes = plt.subplots(
        nrows=6,
        ncols=8,
        figsize=(16, 12),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for i, grid in enumerate(grids):
        for j, noise in enumerate(noises):
            dataset = ConvectionEquationPDEDataset(
                spatial=2 * np.pi,
                temporal=1,
                grid_points=grid,
                snr=noise,
                convection=17,
                seed=17,
            )
            plot_dataset(
                dataset=dataset,
                figure=fig,
                axes=axes,
                params={
                    "title": f"Points: {grid * grid}, Noise: {noise}",
                    "data": {
                        "grid": grid,
                        "extent": [0, 2 * np.pi, 0, 1],
                    },
                    "plot": {
                        "idx_grid": i,
                        "idx_noise": j,
                    },
                },
            )

    # fig.tight_layout()
    fig.suptitle(
        "Noise and (idealized) Sample Size Effect on Convection PDE Dataset",
        fontsize=16,
    )
    plt.savefig(f"./data/img/results_{time.time()}.png")

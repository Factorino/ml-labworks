from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


GRADIENT_CMAP: Colormap = plt.cm.RdBu
POINT_CMAP = ListedColormap(["#FF0000", "#0000FF"])
GRID_STEP = 0.02


def make_grid(
    X: NDArray[np.float64],
    step: float = GRID_STEP,
    margin: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x0_min, x0_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x1_min, x1_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    grid_0, grid_1 = np.meshgrid(
        np.arange(x0_min, x0_max, step),
        np.arange(x1_min, x1_max, step),
    )
    return grid_0, grid_1


def surface_values(
    model: Any,
    grid_0: NDArray[np.float64],
    grid_1: NDArray[np.float64],
) -> NDArray[np.float64]:
    points = np.c_[grid_0.ravel(), grid_1.ravel()]

    if hasattr(model, "decision_function"):
        values = model.decision_function(points)
    elif hasattr(model, "predict_proba"):
        values = model.predict_proba(points)[:, 1]
    else:
        values = model.predict(points)

    return np.asarray(values, dtype=np.float64).reshape(grid_0.shape)


def draw_points(
    ax: Axes,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    alpha: float = 1.0,
    edge: bool = True,
) -> None:
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=POINT_CMAP,
        alpha=alpha,
        edgecolors="k" if edge else "none",
        linewidths=0.5,
    )


def draw_surface(
    ax: Axes,
    model: Any,
    X: NDArray[np.float64],
    *,
    step: float = GRID_STEP,
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int = 21,
) -> None:
    grid_0, grid_1 = make_grid(X, step=step)
    values = surface_values(model, grid_0, grid_1)

    bounds: Any = levels
    if vmin is not None and vmax is not None:
        bounds = np.linspace(vmin, vmax, levels)

    ax.contourf(
        grid_0,
        grid_1,
        values,
        levels=bounds,
        cmap=GRADIENT_CMAP,
        alpha=0.8,
        extend="both",
    )
    ax.set_xlim(grid_0.min(), grid_0.max())
    ax.set_ylim(grid_1.min(), grid_1.max())
    ax.set_xticks(())
    ax.set_yticks(())


def annotate_score(ax: Axes, score: float, *, label: str = "") -> None:
    text = f"{label}{score:.2f}" if label else f"{score:.2f}"
    ax.text(
        0.97,
        0.05,
        text,
        transform=ax.transAxes,
        size=13,
        horizontalalignment="right",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
    )


def class_legend(
    ax: Axes,
    class_names: tuple[str, str] = ("класс 0", "класс 1"),
    sample_names: tuple[str, str] = ("обучающая", "тестовая"),
    *,
    alpha: float = 0.5,
    loc: Any = "upper center",
    anchor: tuple[float, float] = (0.5, -0.02),
    fontsize: int = 8,
) -> None:
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=POINT_CMAP(class_index),
            markeredgecolor="k",
            markeredgewidth=0.5,
            markersize=6,
            alpha=1.0 if sample_index == 0 else alpha,
            label=f"{class_names[class_index]}, {sample_names[sample_index]}",
        )
        for sample_index in (0, 1)
        for class_index in (0, 1)
    ]
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        ncols=2,
        fontsize=fontsize,
        frameon=False,
    )

"""find_coherent_area must grow only through Voronoi cells that touch after
clipping to the survey outline, so a region cannot jump across a survey gap
and its outline is one piece containing the seed."""
import numpy as np
import pytest

pytest.importorskip("shapely")
from shapely.geometry import Point, box  # noqa: E402

import integrate as ig  # noqa: E402


def test_touching_neighbors_drops_separated_cells():
    cells = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1), None]
    neighbors = [[1, 2, 3], [0, 2], [0, 1], [0]]   # as from unclipped Voronoi ridges
    nb = ig.touching_neighbors(neighbors, cells)
    assert nb[0] == [1] and nb[1] == [0]           # shared edge: kept
    assert nb[2] == [] and nb[3] == []             # gap / no cell: dropped


def _u_shaped_survey(step=10.0):
    """Two vertical arms joined at the bottom: the arms' inner edges are
    Voronoi neighbours across the empty inside of the U."""
    xs, ys = np.meshgrid(np.arange(0, 301, step), np.arange(0, 301, step))
    xs, ys = xs.ravel(), ys.ravel()
    keep = (xs <= 50) | (xs >= 250) | (ys <= 50)
    return xs[keep].astype(float), ys[keep].astype(float)


def test_region_does_not_jump_across_gap():
    X, Y = _u_shaped_survey()
    P = np.where(Y >= 100, 1.0, 0.0)               # high only in the upper arms
    A = ig.find_coherent_area(X, Y, P, p_min=0.5, X_center=25.0, Y_center=250.0)
    assert np.all(X[A['idx']] <= 50)                # stays in the left arm
    assert A['polygon'].geom_type == 'Polygon'
    assert A['polygon'].buffer(1e-6).contains(Point(A['X_center'], A['Y_center']))
    assert A['polygon'].area == pytest.approx(A['area'], rel=1e-6)


def test_cells_to_polygon_keeps_seed_part():
    cells = [box(0, 0, 1, 1), box(5, 0, 7, 2)]
    poly = ig.cells_to_polygon(cells, np.array([True, True]), seed=0)
    assert poly.area == pytest.approx(1.0)          # seed part, not the larger one

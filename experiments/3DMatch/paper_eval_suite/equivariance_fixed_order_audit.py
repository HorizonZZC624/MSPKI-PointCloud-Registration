
from __future__ import annotations

import sys
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SUITE_DIR.parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]

for p in (str(EXPERIMENT_DIR), str(PROJECT_ROOT), str(SUITE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import backbone


def _keep_precomputed_neighbor_order(points, neighbor_indices):


    return neighbor_indices


backbone.RotationEquivariantMSPKI._sort_neighbors = staticmethod(
    _keep_precomputed_neighbor_order
)

import equivariance_audit


if __name__ == "__main__":
    equivariance_audit.main()

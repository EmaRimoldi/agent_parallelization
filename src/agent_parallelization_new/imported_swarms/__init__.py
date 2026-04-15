"""Additive port of the cloned agents-swarms blackboard implementation.

This package intentionally coexists with the repository's native
``parallel_shared`` mode. It keeps the imported coordination logic isolated so
that the current BP 2x2 experiments are not replaced or silently changed.
"""

IMPORTED_SWARM_MODE = "imported_swarm"


import os

import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))

INFO_SETS = dict(
    np.load(
        os.path.join(
            current_path,
            "data",
            "info_states.npz",
        )
    )
)


# This is a bit hacky, it allows to transform an infostate into an index, and to
# construct an array to efficiently lookup in Jax.
multiplier = np.array([3**k for k in range(12)])
tr = {k: np.sum((v + 1) * multiplier) % 1235 for k, v in INFO_SETS.items()}
max_value = max(list(tr.values()))
REVERSE_INFO_SETS_LOOKUP = np.zeros(max_value + 1, dtype=int)
REVERSE_INFO_SETS_LOOKUP[list(tr.values())] = np.arange(len(tr))

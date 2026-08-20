import logging
import numpy as np

logger = logging.getLogger(__name__)


def estimate_entropy_floor(inputs, targets, action_spaces, miller_madow=False):
    """
    H(target | observable state), estimated by grouping timesteps that share an
    identical input state. Same normalization as make_ce_loss, so it can be
    subtracted from bc_loss to get KL.

    inputs/targets: dict name -> np.ndarray [T, ...]
    returns: (floors per target, soft_targets per target, mean_group_size)
    """
    X = np.concatenate([v.reshape(len(v), -1) for v in inputs.values()], axis=-1)
    T = len(X)

    keys, group_ids = {}, np.empty(T, dtype=np.int64)
    for i, row in enumerate(X):
        group_ids[i] = keys.setdefault(row.tobytes(), len(keys))
    n_groups = len(keys)
    counts = np.bincount(group_ids, minlength=n_groups)

    floors, soft_targets = {}, {}
    for name, arr in targets.items():
        A = arr.reshape(T, -1)
        sums = np.zeros((n_groups, A.shape[1]))
        np.add.at(sums, group_ids, A)
        p_group = sums / counts[:, None]      # empirical dist per group
        soft_targets[name] = p_group[group_ids]

        h_group, off = np.zeros(n_groups), 0
        for k in action_spaces[name]:
            seg = p_group[:, off:off + k]
            h_group += -(seg * np.log(np.clip(seg, 1e-12, None))).sum(-1)
            if miller_madow:                  # plug-in bias correction, matters for small groups
                h_group += ((seg > 0).sum(-1) - 1) / (2 * counts)
            off += k
        floors[name] = float((h_group * counts).sum() / T)

    return floors, soft_targets, T / n_groups
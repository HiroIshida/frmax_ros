import pickle
from hashlib import md5
from typing import Tuple, Type

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from frmax_ros.hubo_mugcup import MugcupGraspTrainer
from frmax_ros.rollout import AutomaticTrainerBase


def optimize_volume(
    trainer_type: Type[AutomaticTrainerBase], n_iter: int
) -> Tuple[np.ndarray, float]:
    sampler, _ = trainer_type.load_sampler(n_iter)
    pp = trainer_type.get_project_path()
    hash_value = md5(dill.dumps(sampler)).hexdigest()
    cache_path_base = pp / "cache"
    cache_path_base.mkdir(exist_ok=True)
    cache_path = pp / "cache" / f"opt_param-{hash_value}.pkl"
    if not cache_path.exists():
        print(f"Optimizing {hash_value}...")
        best_param = sampler.optimize(200, method="cmaes")
        est_volume = sampler.compute_sliced_volume(best_param)
        with open(cache_path, "wb") as f:
            pickle.dump((best_param, est_volume), f)
        return best_param, est_volume
    else:
        with open(cache_path, "rb") as f:
            best_param, est_volume = pickle.load(f)
        return best_param, est_volume


def get_optimization_history(
    trainer_type: Type[AutomaticTrainerBase],
) -> Tuple[np.ndarray, np.ndarray]:
    _, n_iter = trainer_type.load_sampler()
    param_opt_seq = []
    volume_opt_seq = []

    for i in tqdm.tqdm(range(1, n_iter + 1)):
        param_opt, volume_opt = optimize_volume(trainer_type, i)
        param_opt_seq.append(param_opt)
        volume_opt_seq.append(volume_opt)
    return np.array(param_opt_seq), np.array(volume_opt_seq)


if __name__ == "__main__":
    param_opt_seq, volume_opt_seq = get_optimization_history(MugcupGraspTrainer)
    plt.plot(volume_opt_seq)
    # plt.plot(param_opt_seq[:, :6])
    plt.show()

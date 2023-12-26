import pickle
from hashlib import md5
from typing import Tuple, Type

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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


def visualize_feasible_region(trainer_type: Type[AutomaticTrainerBase], sliced_plot: bool = False):
    sampler, _, _ = trainer_type.load_refined_sampler()

    if sliced_plot:
        fig, ax = plt.subplots()
        n_param = sampler.metric.metirics[0].dim
        axes_slice = list(range(n_param)) + [n_param + 2]
        param_opt = sampler.get_optimal_after_additional()

        colors = [
            "red",
            "orangered",
            "orange",
            "gold",
            "yellow",
            "yellowgreen",
            "greenyellow",
            "green",
        ]
        slice_values = [-0.1, 0.1, 0.3, 0.5, 0.7]

        for i, slice_value in enumerate(slice_values):
            sampler.fslset.show_sliced(
                np.hstack([param_opt, slice_value]), axes_slice, 50, (fig, ax), colors=colors[i]
            )
        plt.show()
    else:
        X = sampler.X[-sampler.count_additional :, -3:]
        Y = np.array(sampler.Y[-sampler.count_additional :])
        X_true = X[Y]
        X_false = X[~Y]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2], c="b")
        ax.scatter(X_false[:, 0], X_false[:, 1], X_false[:, 2], c="r")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("yaw")
        plt.show()


if __name__ == "__main__":
    # param_opt_seq, volume_opt_seq = get_optimization_history(MugcupGraspTrainer)
    # plt.plot(volume_opt_seq)
    # plt.plot(param_opt_seq[:, -8:])
    # plt.show()
    visualize_feasible_region(MugcupGraspTrainer)

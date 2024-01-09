import pickle
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5
from typing import Tuple, Type

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from frmax_ros.hubo_mugcup import GraspingPlanerTrajectory, MugcupGraspTrainer
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


def _optimize_volume(arg) -> Tuple[np.ndarray, float]:
    trainer_type, n_iter = arg
    return optimize_volume(trainer_type, n_iter)


def get_optimization_history(
    trainer_type: Type[AutomaticTrainerBase],
) -> Tuple[np.ndarray, np.ndarray]:
    _, n_iter = trainer_type.load_sampler()

    param_opt_seq = []
    volume_opt_seq = []
    with ProcessPoolExecutor(8) as executor:
        args = [(trainer_type, i) for i in range(1, n_iter + 1)]
        results = list(tqdm.tqdm(executor.map(_optimize_volume, args), total=len(args)))
        for param_opt, volume_opt in results:
            param_opt_seq.append(param_opt)
            volume_opt_seq.append(volume_opt)
    return np.array(param_opt_seq), np.array(volume_opt_seq)


def get_optimal_traj_history(
    trainer_type: Type[AutomaticTrainerBase],
) -> np.ndarray:  # (N, T, 3)
    opt_history = get_optimization_history(trainer_type)
    param_opt_hist, _ = opt_history

    pose_traj_hist = []
    for param in param_opt_hist:
        traj = GraspingPlanerTrajectory(param, dt=0.05, im_using_this_in_demo=False)
        pose_list = []
        for tf in traj.seq_tf_ef_to_nominal:
            pose = tf.to_pose3d()
            pose_list.append(pose)
        pose_traj_hist.append(pose_list)
    return np.array(pose_traj_hist)


def visualize_optimal_traj_history(trainer_type: Type[AutomaticTrainerBase]):
    traj_hist = get_optimal_traj_history(trainer_type)
    n = len(traj_hist)
    planer_traj_list = traj_hist[[0, 50, 150, 250, 350, n - 1]]
    fig, ax = plt.subplots()

    # plot circle patch
    circle = plt.Circle((0, 0), 0.04, color="linen", fill=True, alpha=1.0)
    ax.add_artist(circle)

    # draw rectangle patch
    rect = plt.Rectangle((-0.0075, -0.075), 0.015, 0.04, color="linen", fill=True, alpha=1.0)
    ax.add_artist(rect)

    # draw arrow rigin and x, y axis with color red and blue arrows
    ax.arrow(0, 0, 0.02, 0, head_width=0.005, head_length=0.005, fc="r", ec="r")
    ax.arrow(0, 0, 0, 0.02, head_width=0.005, head_length=0.005, fc="b", ec="g")

    for planer_traj in planer_traj_list:
        x = planer_traj[:, 0]
        y = planer_traj[:, 1]
        yaw = planer_traj[:, 2]
        dx = np.cos(yaw) * 0.006
        dy = np.sin(yaw) * 0.006

        for i in range(len(x)):
            ax.plot([x[i], x[i] + dx[i]], [y[i], y[i] + dy[i]], color="blue", lw=0.5)

        ax.plot(planer_traj[:, 0], planer_traj[:, 1], label="planer", color="gray", lw=0.5)
    ax.set_xlim([-0.075, 0.05])
    ax.set_ylim([-0.08, 0.03])
    ax.set_aspect("equal")
    figure_path = trainer_type.get_project_path() / "figures"
    figure_path.mkdir(exist_ok=True)
    plt.savefig(figure_path / "optimal_traj_history.png", dpi=300)


def visualize_estimated_volume_history(trainer_type: Type[AutomaticTrainerBase]):
    param_opt_seq, volume_opt_seq = get_optimization_history(MugcupGraspTrainer)
    fig, ax = plt.subplots()
    ax.plot(volume_opt_seq)
    ax.set_xlabel("iteration [-]")
    ax.set_ylabel("coverage rate [-]")
    fig.set_size_inches(4, 4)
    plt.tight_layout()
    figure_path = trainer_type.get_project_path() / "figures"
    figure_path.mkdir(exist_ok=True)
    plt.savefig(figure_path / "estimated_volume_history.png", dpi=300)


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

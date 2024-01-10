import pickle
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5
from typing import Tuple, Type

import dill
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

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


def plot_mugcup(ax):
    # body
    circle = plt.Circle((0, 0), 0.04, color="linen", fill=True, alpha=1.0)
    ax.add_artist(circle)

    # handle
    rect = plt.Rectangle((-0.0075, -0.075), 0.015, 0.04, color="linen", fill=True, alpha=1.0)
    ax.add_artist(rect)

    # draw arrow rigin and x, y axis with color red and blue arrows
    ax.arrow(0, 0, 0.02, 0, head_width=0.005, head_length=0.005, fc="r", ec="r")
    ax.arrow(0, 0, 0, 0.02, head_width=0.005, head_length=0.005, fc="b", ec="g")


def plot_planer_traj(ax, planer_traj: np.ndarray, emphasis: bool = False):
    x = planer_traj[:, 0]
    y = planer_traj[:, 1]
    yaw = planer_traj[:, 2]
    dx = np.cos(yaw) * 0.006
    dy = np.sin(yaw) * 0.006

    if emphasis:
        for i in range(len(x)):
            ax.plot([x[i], x[i] + dx[i]], [y[i], y[i] + dy[i]], color="blue", lw=2.0)
        ax.plot(planer_traj[:, 0], planer_traj[:, 1], label="planer", color="black", lw=2.0)
    else:
        for i in range(len(x)):
            ax.plot([x[i], x[i] + dx[i]], [y[i], y[i] + dy[i]], color="blue", lw=0.5)
        ax.plot(planer_traj[:, 0], planer_traj[:, 1], label="planer", color="gray", lw=0.5)


def visualize_optimal_traj_history(trainer_type: Type[AutomaticTrainerBase]):
    traj_hist = get_optimal_traj_history(trainer_type)
    n = len(traj_hist)
    planer_traj_list = traj_hist[[0, 50, 150, 250, 350, n - 1]]
    fig, ax = plt.subplots()

    plot_mugcup(ax)

    for planer_traj in planer_traj_list:
        plot_planer_traj(ax, planer_traj)
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
    n_param = sampler.metric.metirics[0].dim
    param_opt = sampler.get_optimal_after_additional()

    if sliced_plot:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        colors = [
            "red",
            "orange",
            "yellow",
            "greenyellow",
            "green",
        ]
        slice_values = [-0.5, -0.25, 0.0, 0.25, 0.5]

        axes_slice = list(range(n_param)) + [n_param + 2]
        for i, slice_value in enumerate(slice_values):
            label = f"z = {slice_value}"
            sampler.fslset.show_sliced(
                np.hstack([param_opt, slice_value]),
                axes_slice,
                50,
                (fig, ax),
                colors=colors[i],
            )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 0.02)
        ax.set_ylim(-0.03, 0.03)
        plt.show()
    else:
        n = 30
        b_min, b_max = sampler.fslset.b_min[-3:], sampler.fslset.b_max[-3:]
        lins = [np.linspace(l, u, n) for l, u in zip(b_min, b_max)]
        X, Y, Z = np.meshgrid(*lins)
        errors = np.array(list(zip(X.flatten(), Y.flatten(), Z.flatten())))
        param_opt_repeated = np.repeat(param_opt.reshape(1, -1), errors.shape[0], axis=0)
        pts = np.hstack([param_opt_repeated, errors])

        axes_slice = list(range(n_param))
        values = sampler.fslset.func(pts)
        F = values.reshape((n, n, n))
        spacing = (b_max - b_min) / n
        verts, faces, _, _ = measure.marching_cubes(F, 0, spacing=spacing)
        offset = b_min
        verts += offset

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor("k")
        mesh.set_alpha(0.1)
        ax.add_collection3d(mesh)
        ax.set_xlim(-0.05, 0.02)
        ax.set_ylim(-0.03, 0.03)
        ax.set_zlim(-0.6, 0.6)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("yaw [rad]")
        plt.show()


if __name__ == "__main__":
    visualize_optimal_traj_history(MugcupGraspTrainer)
    # visualize_feasible_region(MugcupGraspTrainer, sliced_plot=True)
    # visualize_estimated_volume_history(MugcupGraspTrainer)
    # traj_hist = get_optimal_traj_history(MugcupGraspTrainer)
    # traj = traj_hist[0]
    # visualize_optimal_traj_history(MugcupGraspTrainer)
    # plt.show()
    # param_opt_seq, volume_opt_seq = get_optimization_history(MugcupGraspTrainer)
    # plt.plot(volume_opt_seq)
    # plt.plot(param_opt_seq[:, -8:])
    # plt.show()

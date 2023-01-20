import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from evo.tools import file_interface
from evo.tools import plot
import matplotlib.pyplot as plt
import argparse
# temporarily override some package settings
from evo.tools.settings import SETTINGS

SETTINGS.plot_usetex = False
SETTINGS.plot_axis_marker_scale = 0.1

DROP_FRAMES = True  # drop some keyframes to make the visualization more illusive

def vis_pose(traj_gt, traj_et, name, label0, label1):
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot.PlotMode.xyz)
    traj_by_label = {
        "estimated pose": traj_gt,
        "groundtruth pose": traj_et
    }
    plot.traj(ax, plot.PlotMode.xyz, traj_gt,
              style=SETTINGS.plot_reference_linestyle,
              color=SETTINGS.plot_reference_color, label='groundtruth pose',
              alpha=SETTINGS.plot_reference_alpha)
    plot.traj(ax, plot.PlotMode.xyz, traj_et,
              style=SETTINGS.plot_trajectory_linestyle,
              color='green', label='estimated pose',
              alpha=SETTINGS.plot_reference_alpha)
    # plot.draw_coordinate_axes(ax, traj_gt, plot.PlotMode.xyz,
    #                           SETTINGS.plot_axis_marker_scale)
    # plot.draw_coordinate_axes(ax, traj_et, plot.PlotMode.xyz,
    #                           SETTINGS.plot_axis_marker_scale * 0.1)
    
    fig.axes.append(ax)
    plt.title(name)
    plt.show()

def rotate(traj, gamma=0, beta=0, alpha=0):
    trans = Rotation.from_euler('zyx', [gamma, beta, alpha], degrees=True)
    T = np.eye(4)
    T[:3,:3] = trans.as_matrix()
    print(f"\nT:\n{T}")
    traj.transform(T)
    return traj

def rotate(traj, R):
    T = np.eye(4)
    T[:3,:3] = R
    print(f"\nT:\n{T}")
    traj.transform(T)
    return traj
    
def norm(traj, resize = 1.0):
    track = traj[:,1:4]
    print(f"track: {track}")
    track_norm = np.linalg.norm(np.max(track)) * resize
    print(f"track_norm: {track_norm}")
    track /= track_norm
    print(f"track: {track}")
    traj[:,1:4] = track
    return traj
    
def center(traj):
    T = np.eye(4)
    t = traj.positions_xyz
    T[:3,3] = - t[0]
    traj.transform(T)
    return traj

def match(traj0, traj1, t0, t1):
    idx0 = []
    idx1 = []
    for j, t in enumerate(t1):
        i = np.where(t0 == t)[0]
        if i.size == 1:
            idx0.append(i[0])
            idx1.append(j)
    traj0.reduce_to_ids(idx0)
    traj1.reduce_to_ids(idx1)
    return traj0, traj1

def main(gt_pose, et_pose):
    # timestamp tx ty tz qx qy qz qw
    t_gt = np.genfromtxt(gt_pose)[:,0]
    t_et = np.genfromtxt(et_pose)[:,0]
    
    traj_gt = file_interface.read_tum_trajectory_file(gt_pose)
    traj_et = file_interface.read_tum_trajectory_file(et_pose)
    
    traj_gt, traj_et = match(traj_gt, traj_et, t_gt, t_et)
    
    # num_poses = traj_gt_raw.get_infos()
    # print(f"traj_gt_raw:\n{traj_gt_raw}")
    # print(f"num_poses:\n{num_poses}")
    # print(f"num_poses(0): {num_poses['nr. of poses']}")
    # idx = np.arange(0, num_poses-1, int(num_poses/50))
    
    traj_gt_centered = center(traj_gt)
    # vis_pose(traj_gt, traj_gt_centered, "Groundtruth centered")
    traj_gt_centered.scale(7/100.) 
    # vis_pose(traj_gt, traj_gt_centered, "Groundtruth scaled")
    
    r_a, t_a, s = traj_et.align(traj_gt_centered) 
    traj_et = center(traj_et)
    # print(f"\nr_a:\n{r_a}\nt_a\n{t_a}\ns\n{s}")
    # T_align = traj_et.align_origin(traj_gt_centered)
    # # print(f"\nT_align:\n{T_align}")
    # traj_et = rotate(traj_et, r_a)
    
    vis_pose(traj_gt_centered, traj_et, "Groundtruth vs estimation", "agisoft", "ba trf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SE3 pose.")
    parser.add_argument("--gt_pose", help="ground truth poses in TUM file format.")
    parser.add_argument("--et_pose", help="estimated poses file in TUM format.")
    args = parser.parse_args()
    main(args.gt_pose, args.et_pose)

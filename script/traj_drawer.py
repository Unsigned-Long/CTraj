import argparse

import numpy
import numpy as np
from plt_utils import drawer
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from quaternions import Quaternion


def load_trajectory(filename, down_sample):
    file = open(filename, "r")
    lines = file.readlines()
    content = ''
    for line in lines:
        content += line
    array_buffer = json.loads(content)
    data = []
    for elem in array_buffer['pose_seq']:
        so3_vec = []
        for elem2 in elem['so3']:
            so3_vec.append(elem['so3'][elem2])

        trans_vec = []
        for elem2 in elem['t']:
            trans_vec.append(elem['t'][elem2])
        data.append([so3_vec, trans_vec])
    data_down_sampled = []
    sample = int(1.0 / down_sample)
    for i in range(len(data)):
        if i % sample == 0:
            data_down_sampled.append(data[i])
    return data_down_sampled


def draw_trajectory(ax, data, scale):
    ax.plot(
        [item[1][0] for item in data], [item[1][1] for item in data], [item[1][2] for item in data],
        label='trajectory', c='k', lw=3
    )
    for item in data:
        quat = item[0]
        trans = item[1]
        rot_mat = Quaternion(quat[3], quat[0], quat[1], quat[2]).get_rotation_matrix()
        x_dir = numpy.array(rot_mat).transpose()[0]
        y_dir = numpy.array(rot_mat).transpose()[1]
        z_dir = numpy.array(rot_mat).transpose()[2]
        ax.plot(
            [trans[0], trans[0] + scale * x_dir[0]],
            [trans[1], trans[1] + scale * x_dir[1]],
            [trans[2], trans[2] + scale * x_dir[2]],
            c='r')
        ax.plot(
            [trans[0], trans[0] + scale * y_dir[0]],
            [trans[1], trans[1] + scale * y_dir[1]],
            [trans[2], trans[2] + scale * y_dir[2]],
            c='limegreen')
        ax.plot(
            [trans[0], trans[0] + scale * z_dir[0]],
            [trans[1], trans[1] + scale * z_dir[1]],
            [trans[2], trans[2] + scale * z_dir[2]],
            c='royalblue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Drawer')
    parser.add_argument('filename', type=str, help='the path of the trajectory file to drawer')
    parser.add_argument('--sample', type=float, help='the down sample rate of the pose sequence', default=0.5)
    parser.add_argument('--scale', type=float, help='the scale size of the coordinates', default=0.3)
    args = parser.parse_args()

    traj_filename = args.filename
    down_sample = args.sample
    scale = args.scale

    traj = load_trajectory(traj_filename, down_sample)

    x_min = np.min([elem[1][0] for elem in traj])
    x_max = np.max([elem[1][0] for elem in traj])
    y_min = np.min([elem[1][1] for elem in traj])
    y_max = np.max([elem[1][1] for elem in traj])
    z_min = np.min([elem[1][2] for elem in traj])
    z_max = np.max([elem[1][2] for elem in traj])

    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min

    padding = np.max([x_span, y_span, z_span]) * 0.1

    drawer.set_fig_size(10.0, 8.0)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((x_span + padding * 2, y_span + padding * 2, z_span + padding * 2))

    drawer.set_xticks(ax, x_min - padding, x_max + padding, int(x_span) + 1)
    drawer.set_yticks(ax, y_min - padding, y_max + padding, int(y_span) + 1)
    drawer.set_zticks(ax, z_min - padding, z_max + padding, int(z_span) + 1)
    drawer.set_label_decimal(ax, '%.1f', 'x')
    drawer.set_label_decimal(ax, '%.1f', 'y')
    drawer.set_label_decimal(ax, '%.1f', 'z')

    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    draw_trajectory(ax, traj, scale)

    ax.set_xlabel('X ' + drawer.math_symbols('(m)'))
    ax.set_ylabel('Y ' + drawer.math_symbols('(m)'))
    ax.set_zlabel('Z ' + drawer.math_symbols('(m)'))

    trajectory_patch = mlines.Line2D([], [], c='k', label='trajectory', lw=2)

    ax.legend(handles=[trajectory_patch])
    ax.set_title("Trajectory Represented by Three-Axis Discrete Coordinates")

    drawer.add_grids_3d(ax)
    plt.tight_layout()

    drawer.show_figure()

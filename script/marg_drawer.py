#  Copyright (c) 2023. Created on 10/28/23 1:17 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
#  geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
#  the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
#  systems and multi-sensor fusion.

import argparse
from plt_utils import drawer
from helper import get_array_fields
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
color_name = 'PuOr'
color_range = [0.2, 0.8]
line_alpha = 1
line_width = 2
marg_color = 'gray'
keep_color = 'r'
fig2_color = 'r'


def recovery_mat(data, row, col):
    mat = [[] for elem in range(row)]
    idx = 0
    for v in data.values():
        mat[idx % col].append(v)
        idx += 1
    return mat


def recovery_vec(data):
    vec = []
    for v in data.values():
        vec.append(v)
    return vec


def draw_equ(ax, equation, marg_size, keep_size):
    # mapping
    # equation = equation / np.abs(equation) * np.log10(np.abs(equation) + 1)
    for i in range(len(equation)):
        for j in range(len(equation[i])):
            val = equation[i][j]
            if val > 0.0:
                equation[i][j] = 1
            elif val < 0.0:
                equation[i][j] = -1

    # find value lim
    max = np.max(equation)
    min = np.min(equation)
    val_max = np.max([np.abs(min), np.abs(max)])

    # figure setting

    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    new_colors = plt.get_cmap(color_name)(np.linspace(0, 1, 256))
    new_cmap = ListedColormap(new_colors[int(256 * color_range[0]):int(256 * color_range[1])])
    obj = ax.pcolormesh(np.array(equation), cmap=new_cmap, vmin=-val_max, vmax=val_max)
    plt.colorbar(obj)

    pos = 0
    for elem in marg_size:
        ax.vlines(pos, 0, len(equation), colors=marg_color, alpha=line_alpha, lw=line_width)
        ax.hlines(pos, 0, len(equation), colors=marg_color, alpha=line_alpha, lw=line_width)

        ax.hlines(pos, len(equation) + 1, len(equation) + 2, colors='k', alpha=line_alpha, lw=line_width)
        pos += elem

    mid_pos = pos
    for elem in keep_size:
        ax.vlines(pos, 0, mid_pos, colors=marg_color, alpha=line_alpha, lw=line_width)
        ax.hlines(pos, 0, mid_pos, colors=marg_color, alpha=line_alpha, lw=line_width)

        ax.vlines(pos, mid_pos, len(equation), colors=keep_color, alpha=line_alpha, lw=line_width)
        ax.hlines(pos, mid_pos, len(equation), colors=keep_color, alpha=line_alpha, lw=line_width)

        ax.hlines(pos, len(equation) + 1, len(equation) + 2, colors='k', alpha=line_alpha, lw=line_width)
        pos += elem

    ax.vlines(pos, 0, mid_pos, colors=marg_color, alpha=line_alpha, lw=line_width)
    ax.hlines(pos, 0, mid_pos, colors=marg_color, alpha=line_alpha, lw=line_width)

    ax.vlines(pos, mid_pos, len(equation), colors=keep_color, alpha=line_alpha, lw=line_width)
    ax.hlines(pos, mid_pos, len(equation), colors=keep_color, alpha=line_alpha, lw=line_width)

    ax.hlines(pos, len(equation) + 1, len(equation) + 2, colors=keep_color, alpha=line_alpha, lw=line_width)
    ax.vlines(pos + 1, 0, len(equation), colors='k', alpha=line_alpha, lw=line_width)
    ax.vlines(pos + 2, 0, len(equation), colors='k', alpha=line_alpha, lw=line_width)


def draw_equ_marg(ax, equation, keep_size):
    # mapping
    # equation = equation / np.abs(equation) * np.log10(np.abs(equation) + 1)
    for i in range(len(equation)):
        for j in range(len(equation[i])):
            val = equation[i][j]
            if val > 0.0:
                equation[i][j] = 1
            elif val < 0.0:
                equation[i][j] = -1

    # find value lim
    max = np.max(equation)
    min = np.min(equation)
    val_max = np.max([np.abs(min), np.abs(max)])

    # figure setting

    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    new_colors = plt.get_cmap(color_name)(np.linspace(0, 1, 256))
    new_cmap = ListedColormap(new_colors[int(256 * color_range[0]):int(256 * color_range[1])])

    obj = ax.pcolormesh(np.array(equation), cmap=new_cmap, vmin=-val_max, vmax=val_max)
    plt.colorbar(obj)

    pos = 0
    for elem in keep_size:
        ax.vlines(pos, 0, len(equation), colors=fig2_color, alpha=line_alpha, lw=line_width)
        ax.hlines(pos, 0, len(equation), colors=fig2_color, alpha=line_alpha, lw=line_width)

        ax.hlines(pos, len(equation) + 1, len(equation) + 2, colors='k', alpha=line_alpha, lw=line_width)
        pos += elem
    ax.vlines(pos, 0, len(equation), colors=fig2_color, alpha=line_alpha, lw=line_width)
    ax.hlines(pos, 0, len(equation), colors=fig2_color, alpha=line_alpha, lw=line_width)

    ax.hlines(pos, len(equation) + 1, len(equation) + 2, colors='k', alpha=line_alpha, lw=line_width)
    ax.vlines(pos + 1, 0, len(equation), colors='k', alpha=line_alpha, lw=line_width)
    ax.vlines(pos + 2, 0, len(equation), colors='k', alpha=line_alpha, lw=line_width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Marginalization Drawer')
    parser.add_argument('filename', type=str, help='the path of the marginalization file to drawer')
    args = parser.parse_args()

    marg_filename = args.filename

    data = get_array_fields(marg_filename, ['MarginalizationInfo'])
    margParBlocks = data['margParBlocks']
    keepParBlocks = data['keepParBlocks']

    margParDime = data['margParDime']
    keepParDime = data['keepParDime']

    HMat = data['HMat']
    bVec = data['bVec']

    HMatSchur = data['HMatSchur']
    bVecSchur = data['bVecSchur']

    marg_size = [elem['localSize'] for elem in margParBlocks]
    keep_size = [elem['localSize'] for elem in keepParBlocks]

    drawer.set_fig_size(23.5, 10.0)
    fig, axs = plt.subplots(1, 2)

    H = recovery_mat(HMat, margParDime + keepParDime, margParDime + keepParDime)
    b = recovery_vec(bVec)

    equation = H
    for idx in range(len(b)):
        equation[idx].append(0.0)
        equation[idx].append(b[idx])
    equation = np.array(equation)

    draw_equ(axs[0], equation, marg_size, keep_size)

    H = recovery_mat(HMatSchur, keepParDime, keepParDime)
    b = recovery_vec(bVecSchur)

    equation = H
    for idx in range(len(b)):
        equation[idx].append(0.0)
        equation[idx].append(b[idx])
    equation = np.array(equation)

    draw_equ_marg(axs[1], equation, keep_size)

    drawer.show_figure()

from plt_utils import drawer
from helper import get_array_fields
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

spline = '/home/csl/CppWorks/artwork/ctraj/output/marg/data1/curve.json'
spline_cp = '/home/csl/CppWorks/artwork/ctraj/output/marg/data1/knots.json'
sample_file = '/home/csl/CppWorks/artwork/ctraj/output/marg/samples.json'
fig_output = '/home/csl/CppWorks/artwork/ctraj/output/marg/data1.png'

colors = [
    '#ee1d23', '#3b4ba8', '#231f20', '#b935a2'
]


def load_samples(filename):
    pose_seq = get_array_fields(filename, ['pose_seq'])
    t_ary = []
    for elem in pose_seq:
        t = [elem['t']['r0c0'], elem['t']['r1c0'], elem['t']['r2c0']]
        t_ary.append(t)
    return t_ary


def annotate(ax, x, y, r, text, code):
    # Circle marker
    royal_blue = [0, 20 / 256, 82 / 256]
    c = Circle((x, y), radius=r, clip_on=False, zorder=10, linewidth=2.5,
               edgecolor=royal_blue + [0.6], facecolor='none',
               path_effects=[withStroke(linewidth=7, foreground='white')])
    ax.add_artist(c)

    # use path_effects as a background for the texts
    # draw the path_effects and the colored text separately so that the
    # path_effects cannot clip other texts
    for path_effects in [[withStroke(linewidth=7, foreground='white')], []]:
        color = 'white' if path_effects else royal_blue
        ax.text(x, y - 1.5 * r, text, zorder=100,
                ha='center', va='top', weight='bold', color=color,
                style='italic', fontfamily='monospace',
                path_effects=path_effects)

        # color = 'white' if path_effects else 'black'
        # ax.text(x, y - 2.0 * r, code, zorder=100,
        #         ha='center', va='top', weight='normal', color=color,
        #         fontfamily='monospace', fontsize='medium',
        #         path_effects=path_effects)


def load_spline_cp(filename):
    vel_cp = get_array_fields(filename, ['trajectory', 'pos_spline', 'knots'])
    vel_st = get_array_fields(
        filename, ['trajectory', 'pos_spline', 'start_t'])
    vel_dt = get_array_fields(filename, ['trajectory', 'pos_spline', 'dt'])
    cur_time = vel_st

    vel_knots = []
    vel_knots_time = []
    for elem in vel_cp:
        vel_knots.append([elem['r0c0'], elem['r1c0'], elem['r2c0']])
        vel_knots_time.append(cur_time - vel_dt)
        cur_time += vel_dt

    so3_cp = get_array_fields(filename, ['trajectory', 'so3_spline', 'knots'])
    so3_st = get_array_fields(
        filename, ['trajectory', 'so3_spline', 'start_t'])
    so3_dt = get_array_fields(filename, ['trajectory', 'so3_spline', 'dt'])
    cur_time = so3_st

    so3_knots = []
    so3_knots_time = []
    for elem in so3_cp:
        so3_knots.append([elem['qx'], elem['qy'], elem['qz'], elem['qw']])
        so3_knots_time.append(cur_time - so3_dt)
        cur_time += so3_dt
    return [vel_knots_time, vel_knots, so3_knots_time, so3_knots]


def load_spline(filename, down_sample):
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
        # euler = Quaternion(so3_vec[3], so3_vec[0], so3_vec[1], so3_vec[2]).get_euler()
        # data.append([elem['timeStamp'], np.array(euler) / np.pi * 180.0, trans_vec])
        data.append([elem['timeStamp'], so3_vec, trans_vec])
    data_down_sampled = []
    sample = int(1.0 / down_sample)
    for i in range(len(data)):
        if i % sample == 0:
            data_down_sampled.append(data[i])
    return data_down_sampled


def draw_plot(ax, x, y, color, label=None, alpha=0.6):
    ax.plot(x, y, c=color, lw=4, label=label, alpha=alpha)


def draw_scatter(ax, x, y, color, s=100, label=None):
    ax.scatter(x, y, c=color, label=label, alpha=1, s=s, zorder=100)


def add_arrow(ax, x, y, color):
    for i in range(len(x) - 1):
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.6, lw=2), zorder=200)


if __name__ == '__main__':
    drawer.set_fig_size(10.0, 10.0)
    fig, axs = plt.subplots(1, 1, sharex=True)

    traj = load_spline(spline, 1.0)
    [vel_knots_time, vel_knots, so3_knots_time,
     so3_knots] = load_spline_cp(spline_cp)
    time_seq = [elem[0] for elem in traj]
    quat_seq = [elem[1] for elem in traj]
    vel_seq = [elem[2] for elem in traj]

    samples = load_samples(sample_file)

    draw_plot(axs, [elem[0] for elem in vel_seq], [elem[1] for elem in vel_seq], color=colors[0], label='curve')

    axs.scatter([elem[0] for elem in samples], [elem[1] for elem in samples],
                c='g', label='samples', alpha=0.5, s=40, zorder=100, marker='s')

    draw_scatter(axs, [elem[0] for elem in vel_knots], [elem[1] for elem in vel_knots], color=colors[1], label='knots')
    add_arrow(axs, [elem[0] for elem in vel_knots], [elem[1] for elem in vel_knots], color=colors[1])

    vel_axins = zoomed_inset_axes(axs, zoom=2.5, loc='lower right')
    # fix the number of ticks on the inset axes
    vel_axins.yaxis.get_major_locator().set_params(nbins=7)
    vel_axins.xaxis.get_major_locator().set_params(nbins=7)
    vel_axins.tick_params(labelleft=False, labelbottom=False)

    draw_plot(vel_axins, [elem[0] for elem in vel_seq], [elem[1] for elem in vel_seq], color=colors[0], label='curve')

    vel_axins.scatter([elem[0] for elem in samples], [elem[1] for elem in samples],
                      c='g', label='samples', alpha=0.5, s=40, zorder=100, marker='s')

    draw_scatter(vel_axins, [elem[0] for elem in vel_knots], [elem[1] for elem in vel_knots], color=colors[1],
                 label='knots')
    add_arrow(vel_axins, [elem[0] for elem in vel_knots], [elem[1] for elem in vel_knots], color=colors[1])

    vel_axins.set_xlim(-0.0, 3.0)
    vel_axins.set_ylim(-0.0, 3.0)
    mark_inset(axs, vel_axins, loc1=2, loc2=3,
               fc="r", ec="g", ls='--', lw=3, alpha=0.2)

    axs.legend()
    drawer.set_xticks(axs, -1.0, 17.0, 18)
    drawer.set_yticks(axs, -1.0, 17.0, 18)
    axs.set_xlabel(drawer.math_symbols('x\;(m)'))
    axs.set_ylabel(drawer.math_symbols('y\;(m)'))
    axs.set_aspect(1)
    axs.set_title('Order = 3, Dt = 1, St = 0, Et = 15')
    drawer.add_grids(axs)
    # drawer.add_grids(vel_axins)
    drawer.show_figure(fig_output)

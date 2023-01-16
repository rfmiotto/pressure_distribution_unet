"""
Utility to generate PGF vector files from Python's Matplotlib plots to use in LaTeX documents.

Read more at https://github.com/knly/texfig


Use \showthe\columnwidth to get the figure width in LaTeX


You have 2 options:

1) Manually setup the font style and size from the LaTeX document, generate a
PDF with matplotlib and then import it in the TeX file. In this approach, you
can use `\showthe\font` to determine the font of the document. The output is
something like: \OT1/cmr/m/n/10, which means classic (OT1) font encoded
computer modern roman (cmr) medium weight (m) normal shape (n) 10pt (10) font.
The cmr and other codes can be found in the `texdoc fontname` document. The
problem with this approach is that is it not easy to find and match all the
font styles.

2) Use PGF backend in matplotlib to generate a PGF file and import it into the
TeX document. Generally, however, the compilation time is long and if you have
too many figures, you may get compilation timeout (memory error). A nice 
workaround is to externalize the figure generation: create another TeX
file with all the preamble of the TeX document (the document class, etc). Add
`\thispagestyle{empty}` before `\begin{document}` and put the figure without
caption inside the document content, like:

\thispagestyle{empty}
\begin{document}
% \showthe\font % Use this to determine the font of the figure.
% \showthe\columnwidth % Use this to determine the width of the figure.
\begin{figure}
    \centering
    \inputpgf{./figures/Fig_Cp_Cf_maps_Re60k_M01_M04_periodic}{map_Cp_Cf_60k_M01_M04_periodic.pgf}
\end{figure}
\end{document}

Compile the file and after that, run the command `pdfcrop` (comes with TeXLive) 
in your terminal to crop the resulting PDF file. Now you have an image in a PDF
file that can be readly imported from the main TeX document.
"""

import matplotlib as mpl
mpl.use('pgf')

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)

from shapely.geometry import LineString, Point


width_pt = 469.755 # GET THIS FROM LaTeX using \showthe\columnwidth
width_pt = width_pt*0.97 # multiply by 99% to avoid overfull box
in_per_pt = 1.0/72.27

default_width = width_pt*in_per_pt # in inches
default_ratio = (np.sqrt(5.0) - 1.0) / 2.0 # golden mean ~ 0.618

# You can get the figure width from LaTeX using \showthe\columnwidth

mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.serif": [],
    # "font.serif": ["Times"],
    "font.sans-serif": [],
    "font.monospace": [],
    # 'mathtext.default': 'regular',
    "mathtext.fontset": "cm",
    "font.size": 10,
    "legend.fontsize": 10,
    "axes.labelsize" : 10,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "figure.figsize": [default_width, default_width * default_ratio],
    "axes.linewidth": 0.5,
    "text.latex.preamble": [
        # r"\usepackage{bm}",
        r"\usepackage{amsfonts}",
        r"\usepackage{amssymb}",
        # r"\usepackage{amsmath}",
    ],
    "pgf.preamble": [
        # put LaTeX preamble declarations here
        # r"\usepackage[utf8x]{inputenc}",
        # r"\usepackage[utf8]{inputenc}"
        # r"\usepackage[T1]{fontenc}",
        # macros defined here will be available in plots, e.g.:
        # r"\newcommand{\vect}[1]{#1}",
        # You can use dummy implementations, since you LaTeX document
        # will render these properly, anyway.
    ],
})

import matplotlib.pyplot as plt


def figure(width=default_width, ratio=default_ratio, pad=0, *args, **kwargs):
    """
    Returns a figure with an appropriate size and tight layout.
    """
    fig = plt.figure(figsize=(width, width * ratio), *args, **kwargs)
    # fig.set_tight_layout({
    #     'pad': pad
    # })
    return fig


def subplots(width=default_width, ratio=default_ratio, pad=0, *args, **kwargs):
    """
    Returns subplots with an appropriate figure size and tight layout.
    """
    fig, axes = plt.subplots(figsize=(width, width * ratio), *args, **kwargs)
    # fig.set_tight_layout({
    #     'pad': pad
    # })
    return fig, axes


def savefig(filename, *args, **kwargs):
    """
    Save both a PDF and a PGF file with the given filename.
    """
    plt.savefig(filename + '.pdf', *args, **kwargs)
    plt.savefig(filename + '.pgf', *args, **kwargs)


###############################
#       Customize axis        #
###############################

def make_fancy_axis(ax):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

def major_formatter(x, unit=None):
    if unit == 'degree':
        label = ['{:.2f}'.format(0 if round(val,2)==0 else val).rstrip('0').rstrip('.')+r'$^{\circ}$' for val in x]
        return label
    else:
        return x

def create_parasite_Xaxis(parent_ax, parent_var, new_var, label, ticklabel,
                        type='bottom', pos=-0.25, unit=None):
    ticklabel = [val for val in ticklabel if min(new_var) <= val <= max(new_var)]
    if ticklabel:
        ax = parent_ax.twiny()
        ax.spines[type].set_position(('axes', pos))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines[type].set_visible(True)
        ax.xaxis.set_ticks_position(type)
        ax.xaxis.set_label_position(type)
        ax.spines[type].set_linewidth(0.5)
        ax.xaxis.set_tick_params(width=0.5)
        map = interp1d(new_var, parent_var, kind='nearest')
        axXs = map(ticklabel)
        ax.set_xbound(parent_ax.get_xbound())
        ax.set_xticks(axXs)
        ax.set_xlabel(label, labelpad=1.0)
        ticklabel = major_formatter(ticklabel, unit)
        ax.set_xticklabels(ticklabel)

# def create_parasite_Xaxis(parent_ax, parent_var, new_var, label, ticklabel,
#                         type='bottom', pos=-0.25):
#     # remove ticks outside new_var range
#     ticklabel = [v for v in ticklabel if min(new_var) <= v <= max(new_var)]
#     if not ticklabel:
#         return

#     locations = []
#     values = []
#     first_line = LineString(np.column_stack((parent_var, new_var)))
#     for val in ticklabel:
#         pt1 = Point(parent_var[0], val)
#         pt2 = Point(parent_var[-1], val)
#         second_line = LineString([pt1, pt2])
#         intersection = first_line.intersection(second_line)
#         if intersection.geom_type == 'MultiPoint':
#             x, y = LineString(intersection).xy
#         elif intersection.geom_type == 'Point': 
#             x, y = intersection.xy
#         locations.append(x[0])
#         values.append(y[0])

#     def major_formatter(x):
#         label = ['{:.2f}'.format(0 if round(val,2)==0 else val).rstrip('0').rstrip('.')+r'$^{\circ}$' for val in x]
#         return label
#     values = major_formatter(values)

#     ax = parent_ax.twiny()
#     ax.spines[type].set_position(('axes', pos))
#     ax.set_frame_on(True)
#     ax.patch.set_visible(False)
#     for sp in ax.spines.values():
#         sp.set_visible(False)
#     ax.spines[type].set_visible(True)
#     ax.xaxis.set_ticks_position(type)
#     ax.xaxis.set_label_position(type)
#     ax.spines[type].set_linewidth(0.5)
#     ax.xaxis.set_tick_params(width=0.5)
#     ax.set_xbound(parent_ax.get_xbound())
#     ax.set_xticks(locations)
#     ax.set_xlabel(label, labelpad=1.0)
#     ax.set_xticklabels(values)

def create_parasite_Yaxis(parent_ax, parent_var, new_var, label, ticklabel,
                        type='left', pos=-0.15, unit=None):   
    ticklabel = [val for val in ticklabel if min(new_var) <= val <= max(new_var)]
    if ticklabel:
        ax = parent_ax.twinx()
        ax.spines[type].set_position(('axes', pos))
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines[type].set_visible(True)
        ax.yaxis.set_ticks_position(type)
        ax.yaxis.set_label_position(type)
        ax.spines[type].set_linewidth(0.5)
        ax.yaxis.set_tick_params(width=0.5)
        map = interp1d(new_var, parent_var, kind='nearest')
        axYs = map(ticklabel)
        ax.set_ybound(parent_ax.get_ybound())
        ax.set_yticks(axYs)
        ax.set_ylabel(label, labelpad=0.1)
        ticklabel = major_formatter(ticklabel, unit)
        ax.set_yticklabels(ticklabel)
    else:
        print('No ticklabel found in the range')

def create_parasite_Xaxis_periodic(parent_ax, parent_var, new_var, label, 
                        ticklabel, type='bottom', pos=-0.25):

    # remove ticks outside new_var range
    ticklabel = [v for v in ticklabel if min(new_var) <= v <= max(new_var)]
    if not ticklabel:
        return

    locations = []
    values = []
    first_line = LineString(np.column_stack((parent_var, new_var)))
    for val in ticklabel:
        pt1 = Point(parent_var[0], val)
        pt2 = Point(parent_var[-1], val)
        second_line = LineString([pt1, pt2])
        intersection = first_line.intersection(second_line)
        if intersection.geom_type == 'MultiPoint':
            x, y = LineString(intersection).xy
        elif intersection.geom_type == 'Point': 
            x, y = intersection.xy
        locations.append(x[0])
        locations.append(x[1])
        values.append(y[0])
        values.append(y[1])

    # Forcing 22 deg:
    locations.append((parent_var[-1] - parent_var[0])/2)
    values.append(22)

    # Forcing -6 deg:
    locations.append(parent_var[0])
    values.append(-6)
    locations.append(parent_var[-1])
    values.append(-6)

    def major_formatter(x):
        label = ['{:.2f}'.format(0 if round(val,2)==0 else val).rstrip('0').rstrip('.')+r'$^{\circ}$' for val in x]
        return label
    values = major_formatter(values)

    ax = parent_ax.twiny()
    ax.spines[type].set_position(('axes', pos))
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines[type].set_visible(True)
    ax.xaxis.set_ticks_position(type)
    ax.xaxis.set_label_position(type)
    ax.spines[type].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.set_xbound(parent_ax.get_xbound())
    ax.set_xticks(locations)
    ax.set_xlabel(label, labelpad=1.0)
    ax.set_xticklabels(values)


################################
#   Add image like annotation  #
################################
# OBS: OffsetImage does not seem to work inside AnnotationBbox with 'pgf'
# backend, so that you will have to do that manually..
def add_image(axe, filename, position, size=0.3, head=None, zorder=100,
              fig=plt.gcf(), tail='C', round=True, diag1=0.5, diag2=0.5):
    """ Add an image to the plot
    param axe: main axe of the plot
    param filename: image to be imported
    param size: size of the imported image (in fraction of figure size)
    param head: to which the annotation points (in main axes data coordinates)
    param tail: position of the annotation in the image
    """
    img = plt.imread(filename)
    # Transform position: Data -> Display -> Figure
    pos = fig.transFigure.inverted().transform( (axe.transData.transform(position)) )
    newax = fig.add_axes([pos[0], pos[1], size, size],
                         frameon=False, zorder=zorder, anchor='SW')
    img = newax.imshow(img, zorder=zorder)
    # newax.set_rasterized(True)
    newax.axis('off')
    axpos = newax.get_position() # in figure coordinates
    # transform to data coorinates so that we can use `bbox_inches` when saving
    x0 = axpos.x0 ; x1 = axpos.x1 ; y0 = axpos.y0 ; y1 = axpos.y1
    x0, y0 = fig.transFigure.transform((x0, y0))
    x1, y1 = fig.transFigure.transform((x1, y1))
    x0, y0 = axe.transData.inverted().transform((x0, y0))
    x1, y1 = axe.transData.inverted().transform((x1, y1))
    axpos.x0 = x0 ; axpos.x1 = x1 ; axpos.y0 = y0 ; axpos.y1 = y1
    if head is not None:
        if tail=='C': xytext = (axpos.x0+axpos.x1)/2, (axpos.y0+axpos.y1)/2
        if tail=='N': xytext = (axpos.x0+axpos.x1)/2, axpos.y1
        if tail=='S': xytext = (axpos.x0+axpos.x1)/2, axpos.y0
        if tail=='W': xytext = axpos.x0, (axpos.y0+axpos.y1)/2
        if tail=='E': xytext = axpos.x1, (axpos.y0+axpos.y1)/2
        if tail=='SW': xytext = axpos.x0, axpos.y0
        if tail=='SE': xytext = axpos.x1, axpos.y0
        if tail=='NW': xytext = axpos.x0, axpos.y1
        if tail=='NE': xytext = axpos.x1, axpos.y1
        axe.annotate("",
                    xy=head, xycoords='data',
                    xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle="-|>",
                                    connectionstyle="arc3, rad=0.2",
                                    alpha=0.2,
                                    facecolor='black',
                                    edgecolor='black',
                                    zorder=zorder-1))
    if round is True:
        bbox = newax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        factor = bbox.width / bbox.height
        if diag1 is None or diag2 is None:
            diag1 = diag2 = 0.99*min(newax.transAxes.inverted().transform(
                (newax.get_window_extent().width, newax.get_window_extent().height)))
        patch = patches.Ellipse((0.5, 0.5), diag1, diag2*factor, transform=newax.transAxes,
            edgecolor='black', alpha=0.2, linewidth=1.)
        img.set_clip_path(patch)
        newax.add_patch(patch)

def ticks_in_rad(ax):
    # for ticks in radians (0 to 2pi)
    ax.set_yticks(np.linspace(0, 360, 5))
    ax.set_yticklabels([r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"],
                       rotation='vertical',
                       va='center')
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_locator(MultipleLocator(360/8))

def ticks_in_deg(ax, ticks=[0, 90, 180, 270, 360], minorML=360/12):
    def degrees(y, pos):
        return '{:.0f}'.format(y)+r'$^{\circ}$'
    ax.yaxis.set_minor_locator(MultipleLocator(minorML))
    ax.set_yticks(ticks)
    # ax.set_yticks(np.linspace(0, 360, 5))
    # ax.set_yticklabels([0, 90, 180, 270, 360], rotation='vertical', va='center')
    ax.yaxis.set_major_formatter(FuncFormatter(degrees))
    ax.yaxis.set_tick_params(pad=1)

def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")
    exponent_text = exponent_text.replace("\\times", "")
    return "{} [{} {}]".format(label, exponent_text, units)
    
def format_label_string_with_exponent(ax, axis='both'):  
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(axis=axis, style='sci')
    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw() # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))


def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=0.5, color='k', alpha=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0], 
                                           mpl.lines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(
        arrowstyle=arrowstyle,
        mutation_scale=10*arrowsize,
        color=color,
        alpha=alpha)
    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = patches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows
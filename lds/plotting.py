import seaborn as sb
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import os


# Get current working directory
dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

# Load saved data
entries = os.listdir(dirname + r'/data/simulations/')
for file in entries:
    df1 = pd.read_pickle(dirname + r'/data/simulations/' + file + r'/processed1.pkl')
    df2 = pd.read_pickle(dirname + r'/data/simulations/' + file + r'/processed2.pkl')

    z_label = 'Load Distance Score [-]'
    x1_label = 'Cart 1 X-Coordinates [ft]'
    y1_label = 'Cart 1 Y-Coordinates [ft]'
    x2_label = 'Cart 2 X-Coordinates [ft]'
    y2_label = 'Cart 2 Y-Coordinates [ft]'

    pivoted1 = df1.pivot(y1_label, x1_label, z_label)
    pivoted2 = df2.pivot(y2_label, x2_label, z_label)

    # # Plot heatmap results
    # fig, (ax1, ax2, axcb) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]})
    # ax1.get_shared_y_axes().join(ax2)
    # cmap_reversed = cm.get_cmap('gist_heat_r', 50)
    # g1 = sb.heatmap(pivoted1, cmap=cmap_reversed, ax=ax1, cbar=False)
    # g1.set_ylabel('Cart Y-Coordinate [ft]')
    # g1.set_xlabel(' ')
    # g1.set_title('Cart 1 Floor Position')
    # ax1.invert_yaxis()
    # g2 = sb.heatmap(pivoted2, cmap=cmap_reversed, ax=ax2, cbar_ax=axcb)
    # g2.set_ylabel('')
    # g2.set_xlabel(' ')
    # g2.set_title('Cart 2 Floor Position')
    # g2.set_yticks([])
    # ax2.invert_yaxis()
    # axcb.set_ylabel("Load Distance Score [-]")
    #
    # for ax in [g1, g2]:
    #     tl = ax.get_xticklabels()
    #     ax.set_xticklabels(tl, rotation=60)
    #     tly = ax.get_yticklabels()
    #     ax.set_yticklabels(tly, rotation=0)
    #
    # fig.text(0.5, 0.02, 'Cart X-Coordinate [ft]', ha='center')
    # plt.show()
    #
    # fig.savefig(dirname + r'/plots/heatmap.png')


    # Plot Hospital Surface and 3D Surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = df1[x1_label]
    Y = df1[y1_label]
    Z = df1[z_label]
    ax = fig.add_subplot(111, projection='3d')
    cmap_reversed = cm.get_cmap('gist_heat_r', 50)
    ax.plot_trisurf(X, Y, Z, cmap=cmap_reversed)
    ax.set_xlabel('Cart X-Coordinate [ft]')
    ax.set_ylabel('Cart Y-Coordinate [ft]')
    ax.set_zlabel(z_label)

    img = plt.imread(dirname + r'/plots/FloorLayout.png')
    height, width = img.shape[:2]
    stepX, stepY = 128.0 / width, 124.0 / height
    X1 = np.arange(0, 128, stepX)
    Y1 = np.arange(0, 124, stepY)
    X1, Y1 = np.meshgrid(X1, Y1)
    ax.plot_surface(X1, Y1, np.atleast_2d(0), rstride=1, cstride=1, facecolors=img)
    plt.show()
    fig.savefig(dirname + r'/plots/overlay.png')
    print()

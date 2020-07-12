import seaborn as sb
from matplotlib import cm
import matplotlib.pyplot as plt
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

    # Plot results
    fig, (ax1, ax2, axcb) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]})
    ax1.get_shared_y_axes().join(ax2)
    cmap_reversed = cm.get_cmap('nipy_spectral_r', 100)
    g1 = sb.heatmap(pivoted1, cmap=cmap_reversed, ax=ax1, cbar=False)
    g1.set_ylabel('Cart Y-Coordinate [ft]')
    g1.set_xlabel(' ')
    g1.set_title('Cart 1 Position')
    ax1.invert_yaxis()
    g2 = sb.heatmap(pivoted2, cmap=cmap_reversed, ax=ax2, cbar_ax=axcb)
    g2.set_ylabel('')
    g2.set_xlabel(' ')
    g2.set_title('Cart 2 Position')
    g2.set_yticks([])
    ax2.invert_yaxis()
    axcb.set_ylabel("Load Distance Score [-]")

    for ax in [g1, g2]:
        tl = ax.get_xticklabels()
        ax.set_xticklabels(tl, rotation=45)
        tly = ax.get_yticklabels()
        ax.set_yticklabels(tly, rotation=0)

    fig.text(0.5, 0.02, 'Cart X-Coordinate [ft]', ha='center')
    plt.show()

    fig.savefig(dirname + r'/plots/heatmap.png')
    print('finished')
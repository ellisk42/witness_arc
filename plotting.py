import matplotlib.pyplot as plt                 #support ploting a figure
from matplotlib import colors                   #colors support converting number or argument into colors
import numpy as np

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_arc_array(array):
    cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    n=len(array[0])
    fig, axs = plt.subplots(len(array), n, figsize=(4*n,8), dpi=50, squeeze=False)
    
    
    
    plt.subplots_adjust(wspace=0, hspace=0)

    
    fig_num = 0
    for i, ts in enumerate(array):
        for j, t in enumerate(ts):
            axs[i][j].imshow(t, cmap=cmap, norm=norm)
            axs[i][j].set_yticks(list(range(t.shape[0])))
            axs[i][j].set_xticks(list(range(t.shape[1])))

    plt.tight_layout()


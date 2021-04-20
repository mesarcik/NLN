from matplotlib import pyplot as plt 
from matplotlib.colors import TABLEAU_COLORS as colours
import numpy as np
import pandas as pd 

import sys
sys.path.insert(1,'/home/mmesarcik/NLN/')

colours = [c[4:]  for c in list(colours.keys()) if 'white' not in c]

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N, 0))
    cmap_name = base.name + str(N)
    # edit cvanelteren
    return plt.cm.colors.ListedColormap(color_list, color_list, N)

def main():
    df = pd.read_csv('outputs/test_results.csv')
    df = df.iloc[:-2]#remove the last 2 values

    # plot lines for each model, for recon and NLN -- 
    plt.figure(figsize=(10,10))
    x = np.arange(0,len(pd.unique(df.Class)))

    for i,model in enumerate(pd.unique(df.Model)):
        recon = list(df[(df.Model == model) & (df.NLN == False)].SegmentationAUC)
        nln = list(df[(df.Model == model) & (df.NLN == True)].SegmentationAUC)

        plt.plot(x, recon, label='{} - Recon'.format(model), c=colours[i], marker= 'o')
        plt.plot(x, nln, label='{} - NLN'.format(model), c=colours[i], linewidth = 3, linestyle='--', marker ='*')

        print(' For Model {}, SEG = {}, NLN = {}'.format(model, round(np.mean(recon),3), round(np.mean(nln),3)))

    plt.ylim([0.5,1])
    plt.xticks(np.arange(0,len(pd.unique(df.Class))),pd.unique(df.Class), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/tmp/temp')

if __name__ == '__main__':
    main()

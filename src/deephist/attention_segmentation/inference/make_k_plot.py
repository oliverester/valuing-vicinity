import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
# plot k plot

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})

# rl ={'U-Net + MAF': 
#         {1: 0.5072, 
#          2: 0, 
#          4: 0.5098, 
#          8: 0.5038, 
#         16: 0.5007},
#     'DeepLabV3 + MAF': 
#         {1: 0.5403,
#          2: 0.5513,
#          4: 0.5589,
#          8: 0.5708,
#         16: 0.5504}
#         }

rl ={'DeepLabV3 + MAF': 
        {1: 0.5403,
         2: 0.5513,
         4: 0.5589,
         8: 0.5708,
        16: 0.5504}
        }

df = pd.DataFrame(rl)
df['k'] = df.index

df = df.melt(id_vars='k', var_name='Net')

fig = plt.figure(figsize=(5.1, 3))
p = sns.pointplot(data=df, x='k', y='value', linestyles='dotted')
p.set(ylabel='$DSC_{\mathrm{total}}$', xlabel='Neighbourhood Radius $k$')
#p.legend_.set_title(None)

#fig = p.get_figure()
fig.axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.005))

plt.grid(axis='y')

fig.savefig("k_plot_deeplab.png", bbox_inches = "tight", dpi=400) 
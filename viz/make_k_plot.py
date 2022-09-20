import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
# plot k plot

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 16})




rl ={'DeepLabV3 + MAF': 
        {0: 0.5017,
         1: 0.5502,
         2: 0.5539,
         4: 0.559,
         6: 0.5567,
         8: 0.5671,
         10: 0.5585,
         12: 0.5584,
         14: 0.5599,
         16: 0.5609
         }
}

df = pd.DataFrame(rl)
df['k'] = df.index
df = df.melt(id_vars='k', var_name='Net')

fig = plt.figure(figsize=(5.1, 3))
p = sns.scatterplot(data=df, x='k', y='value')
p = sns.lineplot(data=df, x='k', y='value', linestyle='--')

p.set(ylabel='$\overline{DSC}_{\mathrm{total}}$', xlabel='Neighbourhood radius $k$')
p.set(xticks=[x for x in df['k']], xticklabels=df['k'])

#p.legend_.set_title(None)

#fig = p.get_figure()
fig.axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.01))

plt.grid(axis='y')

fig.savefig("k_plot_deeplab.png", bbox_inches = "tight", dpi=400) 
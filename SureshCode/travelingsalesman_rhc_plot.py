import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(os.path.join('data', 'RHC_tsm.csv'))
sns.set_theme(style="whitegrid")

#fitness v. iters
plt.clf()
sns_plot=sns.lineplot(x="iters", y="fitness", data=df,)
fig=sns_plot.get_figure()
fig.savefig(os.path.join('image', 'RHC_tsm_iter.png'))

#fitness v. fevals
plt.clf()
sns_plot2=sns.lineplot(x="fevals", y="fitness", data=df,)
fig2=sns_plot2.get_figure()
fig2.savefig(os.path.join('image', 'RHC_tsm_fevals.png'))

#fevals/iters v. fitness
plt.clf()
sns_plot3=sns.lineplot(x="fitness", y="value", hue="variable", data=pd.melt(df, ['fitness']), legend="full")
fig3=sns_plot3.get_figure()
fig3.savefig(os.path.join('image', 'RHC_tsm_feval_iter.png'))
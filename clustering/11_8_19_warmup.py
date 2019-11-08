from pydataset import data
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_palette('husl')
mpg = data('mpg')

sns.scatterplot(x='displ', y='hwy',data=mpg, hue = 'cyl')
plt.axhline(mpg.hwy.mean(), color = 'k', linestyle = 'dashed')
plt.axvline(mpg.displ.mean(), color = 'k', linestyle = 'dashed')
plt.title('highway vs. displacement')

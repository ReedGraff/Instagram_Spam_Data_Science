import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib
#from wordcloud import WordCloud

# Univariate analysis of continuous variables
def Side_By_Side():
    # Compile csv data into dataframe
    df = pd.read_csv("Dataset/whole.csv")

    for column in list(df):
        print("Testing: " + column)
        
        # Set initialization info
        fig = plt.figure(figsize=(20,8),facecolor='white')
        gs = fig.add_gridspec(1,2)
        ax = [None for i in range(2)]
        ax[0] = fig.add_subplot(gs[0,0])
        ax[1] = fig.add_subplot(gs[0,1])
        
        # Set Headings
        ax[0].set_title('Distribution of the ' + column + '\n(Kernel Density Estimate)',fontsize=15,fontweight='bold', fontfamily='monospace')
        #ax[0].text(-24,0.019,'Distribution of the ' + column + ' variable',fontsize=23,fontweight='bold', fontfamily='monospace')
        #ax[0].text(-24,0.01826,'Below are the values associated with the ' + column + ' variable.',fontsize=17,fontweight='light', fontfamily='monospace')

        # Set Headings
        ax[1].set_title('Distribution of the ' + column + '\n(Histogram Plot)',fontsize=15,fontweight='bold', fontfamily='monospace')
        #ax[1].text(6,412,'Distribution of the ' + column + ' variable',fontsize=23,fontweight='bold', fontfamily='monospace')
        #ax[1].text(6,395,'Below are the values associated with the ' + column + ' variable.',fontsize=17,fontweight='light', fontfamily='monospace')

        # Kernel Density Estimate
        sns.kdeplot(x=df[column],ax=ax[0],shade=True, color='gold', alpha=1,zorder=3,linewidth=5,edgecolor='black')
        # The y-axis in a density plot is the probability density function for the kernel density estimation.
        
        # Histogram Plot
        sns.histplot(x=df[column],ax=ax[1], color='olive', alpha=1,zorder=2,linewidth=1,edgecolor='black')
        #scatterplot

        for i in range(2):
            ax[i].set_ylabel('')
            ax[i].grid(which='both', axis='y', zorder=0, color='black', linestyle=':', dashes=(2,7))
            
            for direction in ['top','right','left']:
                ax[i].spines[direction].set_visible(False)
                
                
        # Rename and save:
        column = column.translate({ord(i): None for i in '!@#$%^&*(){}[],./ ;:\'\"#'})
        plt.savefig("Visualizations/Visualization_of_" + column + ".png")


Side_By_Side()



"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3])
fig.savefig('test.png')
'''

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

plt.plot(np.random.randn(50).cumsum(), 'k--')
_ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))



#SAVING AND SHOWING 
fig.savefig('test.png')
im = Image.open("test.png")
im.show()
"""

"""
n = range(2,31)

final = 0

for i in n:
    result = ((6*i)-21)
    final += result
    print(result)

print(final)
"""

"""
import numpy as np
from statistics import mode

# reg = [10,15,19,20,22,32,34,37,39,40,45,46]
reg = [4537.3,5660.6, 6428.8, 3029.5, 5785.5, 4853.4, 7045.4, 6965.7, 5040.5, 7266.8, 5043.5, 7257.2, 5716.9, 4937.6, 6590.5]

final = np.mean(reg)
print(final)
final = np.median(reg)
print(final)
final = mode(reg)
print(final)
final = np.std(reg)
final = np.std(reg)
print(final)
"""
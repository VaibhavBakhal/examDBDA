import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pizza =pd.read_csv("C:\\New folder\\advance ana\Datasets\\pizza.csv")
pizza['Promote'].cov(pizza['Sales'])
#variance covaranc
np.cov(pizza['Promote'],pizza['Sales'])# using numpy option
pizza.cov()
#correlation
pizza['Promote'].corr(pizza['Sales'])
pizza['Promote'].corr(pizza['Promote'])#strong +ve
#corelation matrix
pizza.corr()
np.corrcoef(pizza['Promote'],pizza['Sales']) # using numpy option

# scatter plot
sns.scatterplot(data=pizza,x='Promote',y='Sales')
plt.show()
#################################################################

insure =pd.read_csv("C:\\New folder\\advance ana\Datasets\\Insure_auto.csv")
insure['Home'].cov(insure['Home'])
#variance covarance matrix
np.cov(insure['Promote'],insure['Sales'])# using numpy option
insure.cov()
#correlation
#insure['Promote'].corr(insure['Sales'])
#insure['Promote'].corr(insure['Promote'])#strong +ve
#corelation matrix
insure.corr()
np.corrcoef(insure['Promote'],insure['Sales']) # using numpy option

# scatter plot
sns.scatterplot(data=pizza,x='Promote',y='Sales')
plt.show()
#heatmap
sns.heatmap(insure.corr(), xticklabels=insure.corr().columns, yticklabels=insure.corr().columns, annot=True)
plt.show()
sns.pairplot(insure)
plt.show()

iris=pd.read_csv("C:\\New folder\\advance ana\\Datasets\\iris.csv")
print(iris.columns)
iris.cov()
iris.corr()


#heatmap
sns.heatmap(iris.corr(), xticklabels=iris.corr().columns, yticklabels=iris.corr().columns, annot=True)
plt.show()
###########boston

boston=pd.read_csv("C:\\New folder\\advance ana\\Datasets\\boston.csv")
print(boston.columns)
boston.cov()
boston.corr()


#heatmap
sns.heatmap(boston.corr(), xticklabels=boston.corr().columns, yticklabels=boston.corr().columns, annot=False)
plt.show()

sns.pairplot(boston)
plt.show()



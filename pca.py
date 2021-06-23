#loading the data set
import pandas as pd
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df= pd.read_csv(url, names =['sepal length','sepal width', 'petal length', 'petal width', 'target'])
print(df)


from sklearn.preprocessing import StandardScaler

features=['petal length', 'sepal length', 'petal width','sepal width']
x=df.loc[:,'features'].values
y=df.loc[:,['target']].values
x=StandardScaler.fit_transform(x)
print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
	, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf)
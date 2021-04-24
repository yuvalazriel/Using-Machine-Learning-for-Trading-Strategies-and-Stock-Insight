from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import preprocessing
import pandas as pd

# create a pipeline object
pipe = make_pipeline(StandardScaler(),LogisticRegression())
 # load the iris dataset and split it into train and test sets
month = 2
year = 2017
X = pd.read_csv(rf'FinalData\ZscoreForMonth-{str(month)},Year-{str(year)}.csv')
oldY = pd.read_csv(rf'FinalData\classForMonth-{str(month)},Year-{str(year)}.csv')
y = []
for i in range(494):
    y.append(oldY.iloc[i][1])
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# fit the whole pipeline
pipe.fit(X_train, y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),('logisticregression', LogisticRegression())])
# we can now use it like any other estimator
pred=pipe.predict(X_test)
#print(pred)
print(accuracy_score(pred, y_test))
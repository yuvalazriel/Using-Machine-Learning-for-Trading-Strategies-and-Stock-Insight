# importing libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
scaler = StandardScaler()
# Don't cheat - fit only on training data


X_train= pd.read_csv(rf'FINAL DATA\classForYear-2017 to Year-2019.csv')
X_train=X_train.drop(X_train.columns[[0]],1)
print(X_train.head())
y_train= pd.read_csv(rf'FINAL DATA\TrainVectorForYear-2017 to Year-2019.csv')

X_test=pd.read_csv(rf'FINAL DATA\TestZscoreForYear-2020 to Year-2020.csv')
y_test=pd.read_csv(rf'FINAL DATA\TestVectorForYear-2020 to Year-2020.csv')
y_test=y_test.drop(y_test.columns[[0]],1)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=1500)
clf.fit(X_train, y_train.values.ravel())
#MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs',max_iter=1500)
pred=clf.predict_proba(X_test)
#pred=clf.predict(X_test)
'''
columns =['Class']
predict = pd.DataFrame( columns = columns)#, index=y_train.index)
for i in range(X_test.shape[0]):
    predict.iloc[i][0]=pred[i]

predict.to_csv(rf'FINAL DATA\predict.csv',index =False)'''
for i in range(50):
    print(pred[i])

'''
X = pd.read_csv(rf'FINAL DATA\classForYear-2017 to Year-2019.csv')
y_train= pd.read_csv(rf'FINAL DATA\TrainVectorForYear-2017 to Year-2019.csv')
X=X.drop(X.columns[[0]],1)
print(X.head())
# scaling the inputs
scaler = StandardScaler()
X_train= scaler.fit_transform(X)#scaled_X

# Train Test split will be used for both models
#X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
#                                                   test_size = 0.3)

X_test=pd.read_csv(rf'FINAL DATA\TestZscoreForYear-2020 to Year-2020.csv')
y_test=pd.read_csv(rf'FINAL DATA\TestVectorForYear-2020 to Year-2020.csv')
y_test=y_test.drop(y_test.columns[[0]],1)
print(y_test)
# training model with 0.5 alpha value
model = Ridge(alpha = 0.5, normalize = False, tol = 0.001, solver ='auto', random_state = 42)


model.fit(X_train, y_train)

# predicting the y_test
y_pred = model.predict(X_test)

# finding score for our model
score = model.score(X_test, y_test)
print("\n\nModel score : ", score)
'''


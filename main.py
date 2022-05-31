#importing libararies 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
#loading the dataset
df = pd.read_csv('loan_data.csv')
#showing some values and measures of dataset
print('Head ')
print(df.head())
print('Shape : ',df.shape)
print('Describe ',df.describe())
print('NULLS ',df.isnull().sum())
#removing null rows
newdf = df.dropna()
#printing(new df)
df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
#label encoding
label_encoder=LabelEncoder()
df["Loan_ID"]=label_encoder.fit_transform(df["Loan_ID"])
df["Gender"]=label_encoder.fit_transform(df["Gender"])
df["Married"]=label_encoder.fit_transform(df["Married"])
df["Dependents"]=label_encoder.fit_transform(df["Dependents"])
df["Education"]=label_encoder.fit_transform(df["Education"])
df["Self_Employed"]=label_encoder.fit_transform(df["Self_Employed"])
df["Property_Area"]=label_encoder.fit_transform(df["Property_Area"])
df["Loan_Status"]=label_encoder.fit_transform(df["Loan_Status"])
#data scaling
ss = StandardScaler()
df_scaled = ss.fit_transform(df)
df_scaled_df = pd.DataFrame(df_scaled, columns = df.columns)
print('Data after preprocess ')
print(df.describe())

#drop columns and 1 means drop column
x= df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = df["Loan_Status"]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=False,random_state=0)
# Data shape before and after splitting
print('x shape , x tarin , y test ',x.shape, x_train.shape, x_test.shape)
print('y shape , y tarin , y test ',y.shape,y_train.shape,y_test.shape)
#calculating SVM accuracy
model = svm.SVC()
model.fit(x_train, y_train)
aa= model.predict(x_test)
accuracy_svm = metrics.accuracy_score(y_test,aa)
print('SVM Accuracy:',accuracy_svm)
#Decision Tree Classification
model2 = DecisionTreeClassifier(criterion='entropy',random_state=0)
model2.fit(x_train, y_train)
y_predict = model2.predict(x_test)
accuracy_DecisionTree = metrics.accuracy_score(y_test,y_predict)
print('Decision Tree Classifier Accuracy:',accuracy_DecisionTree)
#Logistic_regression
model5 = LogisticRegression(solver = "liblinear" ,random_state=0)
model5.fit(x_train , y_train)

lr_pred= model5.predict (x_test)

lr_accuracy = metrics.accuracy_score(y_test,lr_pred)
print('Accuracy of logistic regression :' ,lr_accuracy)

#PCA feature extraction
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(df_scaled)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
x_pca=pca.transform(df_scaled)
print(df_scaled.shape)
print(x_pca.shape)

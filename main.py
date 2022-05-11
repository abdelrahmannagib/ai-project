import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df = pd.read_csv('loan_data.csv')

#removing null rows
newdf = df.dropna()

#print(newdf)
# bondo2
df.dropna(inplace = True)

df.drop_duplicates(inplace = True)

df.dtypes

label_encoder=LabelEncoder()

df["Loan_ID"]=label_encoder.fit_transform(df["Loan_ID"])

df["Gender"]=label_encoder.fit_transform(df["Gender"])

df["Married"]=label_encoder.fit_transform(df["Married"])

df["Dependents"]=label_encoder.fit_transform(df["Dependents"])

df["Education"]=label_encoder.fit_transform(df["Education"])

df["Self_Employed"]=label_encoder.fit_transform(df["Self_Employed"])

df["Property_Area"]=label_encoder.fit_transform(df["Property_Area"])

df["Loan_Status"]=label_encoder.fit_transform(df["Loan_Status"])
#print(df.describe())
#print(newdf)
#drop columns and 1 means drop column
x= df.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = df["Loan_Status"]
#print (y)
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
# Data shape before and after splitting
print(x.shape, x_train.shape, x_test.shape)

model = svm.SVC(random_state=0)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print('SVM Accuracy:',accuracy)

#Decision Tree Classification

model2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
model2.fit(x_train, y_train)

y_predict = model2.predict(x_test)

accuracy_DecisionTree = metrics.accuracy_score(y_test,y_predict)

print('Decision Tree Classifier Accuracy:',accuracy_DecisionTree)








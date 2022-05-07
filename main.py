import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
print(newdf)

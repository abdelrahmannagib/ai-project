import pandas as pd

df = pd.read_csv('loan_data.csv')

#removing null rows
newdf = df.dropna()

print(newdf)

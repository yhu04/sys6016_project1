import pandas as pd
from sklearn import preprocessing



column_names = ["pclass", "survived", "name", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked", "boat", "body", "home.dest"]

def impute_data(data):
	imputer = preprocessing.Imputer(missing_values='NaN',strategy='median')
	# imputed_data = imputer.fit_transform(data)

	for name in column_names:
		if data[name].dtype != "object": #unclear how to impute strings...
			imputed_data = imputer.fit_transform(data[name].reshape(-1, 1))
			data[name] = imputed_data
			print data[name]
	return data

def get_data(path):
	return pd.read_excel(path, 'titanic3')



if __name__ == "__main__":
	path = "../../data/titanic3.xls"
	data = get_data(path)
	print impute_data(data)

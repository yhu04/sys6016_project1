
from utils.plotting import plot_dataframe_attribute
from utils.preprocessing import get_data, impute_data

if __name__ == "__main__":

	data = get_data("../data/titanic3.xls")
	imputed_data = impute_data(data)
	plot_dataframe_attribute(imputed_data, "age", ["mean", "std"])
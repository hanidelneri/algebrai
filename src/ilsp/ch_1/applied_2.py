import pandas as pd
import matplotlib.pyplot as plt


autos = pd.read_csv("Auto.csv")
print(autos.describe())

dropped_row_autos = autos.drop(autos.index[10:86])
print(dropped_row_autos.describe())

pd.plotting.scatter_matrix(autos)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

college = pd.read_csv("College.csv")
college.head(10)
college.dtypes

college_with_index = pd.read_csv("College.csv", index_col=0)
college_with_index

college_with_index_renamed = college.rename({"Unnamed: 0": "College"}, axis=1)
college_with_index_renamed = college_with_index_renamed.set_index("College")
college_with_index_renamed.head(10)

college_with_index_renamed.describe()
# pd.plotting.scatter_matrix(college_with_index_renamed[["Top10perc", "Apps", "Enroll"]])


# college_with_index_renamed.boxplot(column=["Outstate"])
# plt.show()

college_with_index_renamed["Elite"] = college_with_index_renamed["Top10perc"] > 50
college_with_index_renamed["Elite"].sum()

# college_with_index_renamed.boxplot(column=["Outstate"], by="Elite")
# plt.show()

college.hist(bins=10, figsize=(10, 8), grid=False)
plt.tight_layout()
plt.show()

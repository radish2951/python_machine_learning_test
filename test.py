from sklearn import linear_model
import pandas as pd
import numpy as np

colors = pd.read_csv("colors.csv", sep=",")
#colors.head()

colors["rr"] = colors["r"] ** 2
colors["gg"] = colors["g"] ** 2
colors["bb"] = colors["b"] ** 2
colors["rg"] = colors["r"] * colors["g"]
colors["gb"] = colors["g"] * colors["b"]
colors["br"] = colors["b"] * colors["r"]
colors["rrr"] = colors["rr"] * colors["r"]
colors["rrg"] = colors["rr"] * colors["g"]
colors["rrb"] = colors["rr"] * colors["b"]
colors["rgg"] = colors["rg"] * colors["g"]
colors["rgb"] = colors["rg"] * colors["b"]
colors["rbb"] = colors["br"] * colors["b"]
colors["ggg"] = colors["gg"] * colors["g"]
colors["ggb"] = colors["gg"] * colors["b"]
colors["gbb"] = colors["gb"] * colors["b"]
colors["bbb"] = colors["bb"] * colors["b"]

#colors2 = colors.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
#colors2.head()

clf_r = linear_model.LinearRegression()
clf_g = linear_model.LinearRegression()
clf_b = linear_model.LinearRegression()

# X = wine.loc[:, ['density']].as_matrix()
# y = wine['alcohol'].as_matrix()

print(colors[0:10])

X = colors.drop(["answer_r", "answer_g", "answer_b"], axis=1).as_matrix()

y_r = colors["answer_r"].as_matrix()
y_g = colors["answer_g"].as_matrix()
y_b = colors["answer_b"].as_matrix()

clf_r.fit(X, y_r)
clf_g.fit(X, y_g)
clf_b.fit(X, y_b)

print(clf_r.coef_)
print(clf_g.coef_)
print(clf_b.coef_)

print(clf_r.intercept_)
print(clf_g.intercept_)
print(clf_b.intercept_)

# print(clf.score(X, y))

# plt.scatter(X, y)

# plt.plot(X, clf.predict(X))

# plt.show()

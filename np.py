from sklearn import linear_model
import numpy as np

clf = linear_model.LinearRegression()

X = [[1, 2],
     [3, 7],
     [7, 10],
     [15, 17],
     [4, 10]
    ]

y = [[1],
     [2],
     [3],
     [4],
     [5]
    ]

clf.fit(X, y)

print(clf.coef_)
print(clf.intercept_)

import numpy as np
import matplotlib.pyplot as plt
from random_forest_isolation import RandomForestIsolation
from mylogg import mylogger
import logging

rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2) # synthetic dataset, two Gaussians
X_train = np.r_[X + 2, X - 2]
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Xgrid = np.c_[xx.ravel(), yy.ravel()] # where to compute the score
ratio_samples, num_trees, num_random_features, do_multiprocessing, extra_trees = 0.5, 100, 1, False, 'none'
iso = RandomForestIsolation(ratio_samples, num_trees, num_random_features,do_multiprocessing,extra_trees)
iso.fit(X_train)
scores = iso.predict(Xgrid)
print("scores: ", scores)
scores = scores.reshape(xx.shape)
plt.title("IsolationForest")
m = plt.contourf(xx, yy, scores, cmap=plt.cm.Blues_r)
plt.colorbar(m)
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=20, edgecolor="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="green", s=20, edgecolor="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="red", s=20, edgecolor="k")
plt.axis("tight"), plt.xlim((-5, 5)), plt.ylim((-5, 5))
plt.legend( [b1, b2, c], ["training observations","new regular observations","new abnormal observations"], loc="upper left")
plt.show(block=True) 
# PGM 1 [A]
import pandas as pd
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

df = pd.read_csv('/content/placement-dataset.csv')

X = df[['cgpa', 'iq']]
y = df['placement']

model = Perceptron()
model.fit(X, y)

print("Weights:", model.coef_)
print("Bias:", model.intercept_)

print("Accuracy:", model.score(X, y))

plot_decision_regions(X.values, y.values, clf=model, legend=1)
plt.show()
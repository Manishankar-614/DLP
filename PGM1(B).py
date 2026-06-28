#PGM 1 [B]
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

def train_and_print(data, name):
    X = data[['input1', 'input2']]
    y = data['output']

    model = Perceptron()
    model.fit(X, y)

    print(f"{name} Weights:", model.coef_)
    print(f"{name} Bias:", model.intercept_)

    return model, X, y

data = {
    "AND": pd.DataFrame({'input1':[1,1,0,0],'input2':[1,0,1,0],'output':[1,0,0,0]}),
    "OR":  pd.DataFrame({'input1':[1,1,0,0],'input2':[1,0,1,0],'output':[1,1,1,0]}),
    "XOR": pd.DataFrame({'input1':[1,1,0,0],'input2':[1,0,1,0],'output':[0,1,1,0]})
}

models = {}
for name, df in data.items():
    models[name] = train_and_print(df, name)

model, X, y = models["XOR"]
plot_decision_regions(X.values, y.values, clf=model, legend=2)
plt.show()
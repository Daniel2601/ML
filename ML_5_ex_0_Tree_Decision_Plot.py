print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

# Chargement des donnees iris
iris = load_iris()

# Préparation des données d'entrainement uniquement sur les critères de sépales
X = iris.data[:, [0, 1]]
y = iris.target

# Création du modèle à partir des données d'entrainement X et y
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3).fit(X, y)

# Visualisation de l'arbre
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
...                      feature_names=iris.feature_names[0:2],  
...                      class_names=iris.target_names,  
...                      filled=True, rounded=True,  
...                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris2") 


plot_step = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Visualisation des frontières de décision
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Affichage des points d'entrainement
n_classes = 3
plot_colors = "ryb"
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()





# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:51:48 2018

@author: diaae
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# Simple Linear Regression


class LinearRegression(object):
    def fit(self, data):
        self.X_train, self.y_train = data[0], data[1]

        # Fitting Simple Linear Regression to the Training set        
        from sklearn.linear_model import LinearRegression as lr
        self.regressor = lr()
        self.regressor.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting the Test set results
        y_pred = self.regressor.predict(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', point_color='green', line_color='blue'):
        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Training set & Test set results
        plt.scatter(X, y, color=point_color)
        plt.plot(X, self.regressor.predict(X), color=line_color)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()


'''

multiple regression

'''


# Polynomial Regression
class PolynomialRegression(object):

    def fit(self, data, degree=2):

        self.X_train, self.y_train = data[0], data[1]

        # Fitting Polynomial Regression to the dataset
        from sklearn.linear_model import LinearRegression as lr
        from sklearn.preprocessing import PolynomialFeatures as pf

        self.poly_reg = pf(degree)
        X_poly = self.poly_reg.fit_transform(self.X_train)
        self.lin_reg = lr()
        self.lin_reg.fit(X_poly, self.y_train)

    def predict(self, test):
        # Predicting the Test set results
        test = self.poly_reg.transform(test)
        y_pred = self.lin_reg.predict(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', point_color='green', line_color='blue'):
        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Polynomial Regression results
        plt.scatter(X, y, color=point_color)
        plt.plot(X, self.predict(y), color=line_color)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()


'''

SCR

'''


# Decision Tree Regression
class DecisionTreeRegression(object):

    def fit(self, data, random_state=0):

        self.X_train, self.y_train = data[0], data[1]

        # Fitting Decision Tree Regression to the dataset
        from sklearn.tree import DecisionTreeRegressor as dtr
        self.regressor = dtr(random_state=random_state)
        self.regressor.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting a new result
        y_pred = self.regressor(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):
        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Decision Tree Regression results (higher resolution)  
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color=train_point_color)
        plt.plot(X_grid, self.regressor.predict(X_grid), color=line_color)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()


# Random Forest Regression

class RandomForestRegression(object):

    def fit(self, data, n_estimators=10, random_state=0):

        self.X_train, self.y_train = data[0], data[1]

        # Fitting Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor as rfr
        self.regressor = rfr(n_estimators=n_estimators, random_state=random_state)
        self.regressor.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting a new result
        y_pred = self.regressor.predict(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Random Forest Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color=train_point_color)
        plt.plot(X_grid, self.predict(X_grid), color=line_color)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()


# Logistic Regression

class LogisticRegression(object):

    def fit(self, data, test_size=0.2, random_state=0):

        self.X_train, self.y_train = data[0], data[1]

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)

        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression(random_state=0)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting the Test set results
        y_pred = self.classifier.predict(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X, y
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()


# K-Nearest Neighbors (K-NN)


class KNearestNeighbors(object):

    def fit(self, data, n_neighbors=5, metric='minkowski', p=2):

        self.X_train, self.y_train = data[0], data[1]

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)

        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting the Test set results
        y_pred = self.classifier.predict(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X, y
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()


# Support Vector Machine (SVM)


class SupportVectorMachine(object):

    def fit(self, data, test_size=0.2, kernel='linear', metric='minkowski', p=2, random_state=0):

        self.X_train, self.y_train = data[0], data[1]

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)

        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        self.classifier = SVC(kernel=kernel, random_state=0)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting the Test set results
        y_pred = self.classifier.predict(test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X, y
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('SVM (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()


# Random Forest Classification

class RandomForestClassification(object):

    def fit(self, data, test_size=0.2, n_estimators=10, criterion='entropy', random_state=0):

        self.X_train, self.y_train = data[0], data[1]

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)

        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                                 random_state=random_state)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, test):
        # Predicting the Test set results
        y_pred = self.classifier.predict(self.X_test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
            y = self.y_train
        else:
            X = data[0]
            y = data[1]

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X, y
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, self.classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()


#        

# K-Means Clustering

class KMeans(object):

    def fit(self, data, test_size=0.2, n_estimators=10, criterion='entropy', random_state=0):

        self.X_train = data

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)

        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                                 random_state=random_state)

    def optimal_number_of_clusters(self, x=(0, 11), title, xLable='x', yLabel='y'):
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(x[0], x[1]):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(self.X_train)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title(title)
        plt.xlabel(xLable)
        plt.ylabel(yLabel)
        plt.show()

    def predict(self, test):
        # Predicting the Test set results
        y_pred = self.classifier.predict(self.X_test)
        return y_pred

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
        else:
            X = data

        from sklearn.cluster import KMeans

        # Fitting K-Means to the dataset
        kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(self.X_train)

        # Visualising the clusters
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()


# Hierarchical Clustering

class Hierarchical(object):

    def fit(self, data, test_size=0.2, n_estimators=10, criterion='entropy', random_state=0):

        self.X_train, self.y_train = data[0], data[1]

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)

        # Fitting Hierarchical Clustering to the dataset
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        self.y_hc = hc.fit_predict(self.X_train)

    def dendrogram(self, method='ward', title, xLable='x', yLabel='y'):
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(self.X_train, method=method))
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

    def display(self, title, data=None, xLabel='X', yLabel='Y', train_point_color='green', test_point_color='red',
                line_color='blue', predict=None, predict_color='brown'):

        if data is None:
            X = self.X_train
        else:
            X = data

        y_hc = self.y_hc
        # Visualising the clusters
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend()
        plt.show()



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

class IdentityScaler(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X

class PathMachine(BaseEstimator):

    def __init__(self, norm=np.linalg.norm, classify=False):
        self.norm = norm
        self.classify = classify
        self.x_scaler = IdentityScaler() if self.classify else StandardScaler()
        self.y_scaler = IdentityScaler() if self.classify else StandardScaler()

    def find_start(self, X):
        index_max = None
        value_max = -np.inf
        for index, x in enumerate(X):
            value = self.norm(x)
            if value > value_max:
                index_max = index
                value_max = value
        return index_max

    def find_next(self, point, target, X, y):
        index_min = None
        value_min = np.inf
        for index, x in enumerate(X):
            if self.classify and (y[index] != target):
               continue
            value = self.norm(x - point)
            if value < value_min:
                index_min = index
                value_min = value
        return index_min

    def fit(self, X, y):
        X = np.copy(X)
        X = self.x_scaler.fit_transform(X)

        y = np.copy(y).reshape(-1, 1)
        y = self.y_scaler.fit_transform(y).flatten()

        self.paths = {} if self.classify else []

        start_index = self.find_start(X)
        start_value = X[start_index]
        start_target = y[start_index]

        X = np.delete(X, start_index, axis=0)
        y = np.delete(y, start_index, axis=0)

        while len(X) > 0:
            next_index = self.find_next(start_value, start_target, X, y)
            if self.classify and next_index is None:
                start_index = self.find_start(X)
                start_value = X[start_index]
                start_target = y[start_index]
                continue
            next_target = y[next_index]

            if self.classify:
                if not next_target in self.paths:
                    self.paths[next_target] = []
                self.paths[next_target].append({
                    'start': start_value,
                    'next': X[next_index]
                })
            else:
                self.paths.append({
                    'start': start_value,
                    'next': X[next_index],
                    'value': start_target,
                    'target': next_target
                })

            start_value = X[next_index]
            start_target = y[next_index]

            X = np.delete(X, next_index, axis=0)
            y = np.delete(y, next_index, axis=0)

        return self

    def predict(self, X):
        result = []
        X = self.x_scaler.transform(np.copy(X))

        for x in X:
            if self.classify:
                predicted = None
                min_distance = np.inf
                for target in self.paths:
                    for path in self.paths[target]:
                        point = x - path['start']
                        line = path['next'] - path['start']
                        if np.allclose(self.norm(line), 0):
                            continue
                        direction = line / self.norm(line)
                        product = np.dot(point, direction)
                        projection = product * direction

                        distance = self.norm(projection - point)
                        if distance < min_distance:
                            predicted = target
                            min_distance = distance
                result.append(predicted)
            else:
                predicted = None
                min_distance = np.inf
                for path in self.paths:
                    point = x - path['start']
                    line = path['next'] - path['start']
                    if np.allclose(self.norm(line), 0):
                            continue
                    direction = line / self.norm(line)
                    product = np.dot(point, direction)
                    projection = product * direction
                    parameter = np.sign(product) * self.norm(projection) /\
                               self.norm(line)

                    distance = self.norm(projection - point)
                    if distance < min_distance:
                        predicted = (1 - parameter) * path['value'] +\
                                   parameter * path['target']
                        min_distance = distance
                result.append(predicted)

        return self.y_scaler.inverse_transform(
            np.array(result).reshape(-1, 1)
        ).flatten()

    def score(self, X, y):
        if self.classify:
            return f1_score(y.flatten(), self.predict(X), average='micro')
        else:
            return r2_score(y.flatten(), self.predict(X))


import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a path machine classifier
classifier = PathMachine(classify=True)

print("Start to fit")

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

print("Start predict")

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

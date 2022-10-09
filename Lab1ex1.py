from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

target = digits.target # target numbers from 0 to 9
data = digits.data # matrix with (n smaples, n features)

image_digit0 = digits.images[0]

# print(image_digit0)

# Prepare training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2, random_state = 0)
print("We have {} training samples, corresponding to features, and {} test samples, corresponding to numbers".format(len(y_train), len(y_test)))

# we use an SVM estimator and feed it with the data
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# calculate accuracy score
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, y_pred, normalize=True)
n_correct = accuracy_score(y_test, y_pred, normalize=False)
print("Number of correctly classified samples is {}, total number of samples is {}.".format(n_correct, len(y_test)))
print("The test accuracy, defined as number of correctly classified samples over total number of test samples, is {:.2f}%".format(test_acc*100))

########## CONFUSION MATRIXES
# create a confusion matrix 
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.utils.multiclass import unique_labels
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=digits.target_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=digits.target_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# let´s see one of the test images
print("We can see that the shape of the sample data is {}".format(X_test[-1:].shape))

# We resize it using numpy resize function:
test_image = np.resize(X_test[-1:], (8,8))
print("Now we have the image shape {}".format(test_image.shape))

# Let's visualize the image:
plt.imshow(test_image)

print("The test image was number {} and the predicted one {}".format(y_test[-1:], y_pred[-1:]))
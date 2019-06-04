from sklearn.datasets import fetch_mldata
from sklearn import neighbors
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import os

def getKnn(xtrain, ytrain, xtest, ytest, k):
    # create knn
    knn = neighbors.KNeighborsClassifier(k)
    # train
    knn.fit(xtrain, ytrain)
    # verify error
    error = 1 - knn.score(xtest, ytest)
    print('Try with k = ' + str(k) + ' Erreur: %f' % error)

    return knn

#we can try to find best
def find_best_k(xtrain, ytrain, xtest, ytest):
    # try to fnd best K
    errors = []
    best_k = 0;
    best_score = 0;
    for k in range(2, 10):
        knn = neighbors.KNeighborsClassifier(k)
        # get error in %
        score = knn.fit(xtrain, ytrain).score(xtest, ytest)
        print("score for k =" + str(k) + " : " + str(score))
        if  score > best_score :
            best_score = score
            best_k = k

    print("Best k is :" + str(best_k) + " with a score :" + str(best_score))
    return best_k

def show_random_sample_result(knn, xtest):
    # get sample of result prediction
    predicted = knn.predict(xtest)
    # get good format
    images = xtest.reshape(-1, 28, 28)
    # select 12 random result
    res = np.random.randint(images.shape[0], size=12)
    # On affiche les images avec la prédiction associée
    for index, value in enumerate(res):
        plt.subplot(3, 4, index + 1)
        plt.axis('off')
        plt.imshow(images[value], cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title('Predicted: %i' % predicted[value])

    plt.show()

def show_random_bad_prediction(knn, xtest):
    # get data bad prediction
    predicted = knn.predict(xtest)
    images = xtest.reshape(-1, 28, 28)
    res = np.random.randint(images.shape[0], size=12)

    misclass = (ytest != predicted)
    misclass_images = images[misclass, :, :]
    misclass_predicted = predicted[misclass]

    # on sélectionne un échantillon de ces images
    res = np.random.randint(misclass_images.shape[0], size=12)

    # on affiche les images et les prédictions (erronées) associées à ces images
    for index, value in enumerate(res):
        plt.subplot(3, 4, index + 1)
        plt.axis('off')
        plt.imshow(misclass_images[value], cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title('Predicted: %i' % misclass_predicted[value])

    plt.show()


# fetch dataset, you must have  mldata/mnist-original.mat in your working directory
# mnist.data.shape, dataset with images
# mnist.data.target, dataset with correspondig value
mnist = fetch_mldata('MNIST original', data_home=os.getcwd())

# sampling dataset, choose 5000 random value
# better to use,     sklearn.utils.resample to avoid duplicate
sample = np.random.randint(70000, size=15000)
data = mnist.data[sample]
target = mnist.target[sample]

#split data, between a training set and a testing set
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)
k = find_best_k(xtrain, ytrain, xtest, ytest)
knn = getKnn(xtrain, ytrain, xtest, ytest, k)
#show_random_sample_result(knn,xtest)
show_random_bad_prediction(knn,xtest)
"""
mnist_svm
~~~~~~~~~

https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/mnist_svm.py

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
#import mnist_loader

# Third-party libraries
from sklearn import svm
import cPickle, gzip, numpy

def svm_baseline():
    f = gzip.open('/Users/parrt/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)

    # train
    clf = svm.SVC()
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))

if __name__ == "__main__":
    svm_baseline()

"""
I got output:
Baseline classifier using an SVM.
9435 of 10000 values correct.
"""

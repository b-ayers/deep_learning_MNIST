import numpy as np
import gzip
import _pickle as cPickle
from neural_network import *

#load data from MNIST database for training and testing of Neural_Network
#50000 training sets and 10000 testing sets

#######LOAD IN DATA########
#code to read in MINST data and format into batch size matrices
#data pulled from Neilson's file, going to reformat it myself
f = gzip.open('data/mnist.pkl.gz', 'rb')
imported = cPickle.Unpickler(file=f, encoding='latin1')
training_data, validation_data, test_data = imported.load()
f.close()
#training_data is a tuple with the first entry being a numpy array with 50,000 entries,
#each of which is an array of 784 values for each 28x28 image
train = training_data[0].T
train_sol = training_data[1]
test = test_data[0].T



########PLAY WITH TOYS#######
#choosing different step size, batch size, and layer structure has widely varying results
#as well as effects of random weight initialization, some hands are better than others
#need to learn more about optimizing these parameters
#MAKE SURE for now that 5000 is evenly divisible by chosen batch size
network = Neural_Network([784, 200, 100, 10], 10)
#if loading a file, make sure you initialize the network to have same layers
#network.load_weights('98.4_save_784_300_200_100_10_step_10.0_batch_10_weights.pkl', '98.4_save_784_300_200_100_10_step_10.0_batch_10_biases.pkl')
#network.load_weights('98.47save_784_300_200_100_10_from98.13_weights.pkl', '98.47save_784_300_200_100_10_from98.13_biases.pkl')

#how many times to cycle through the full training set
cycles = 1
#step size, can influence results dramatically, needs to be adjusted when changing other parameters
st = 10.0
testsize = 10000
percent = 0.0

for c in range(0, cycles):
    #to shuffle both the solutions and inputs in the same order
    #questionable efficiency
    permute = np.random.permutation(50000)
    train = train[:, permute]
    train_sol = train_sol[permute]
    #was previously only using
    #r = np.random.permutation(range(0, 50000, network.batch))
    #but realized I was taking the same slices, only in different orders

    #starting points separated by batchsize
    r = range(0, 50000, network.batch)
    for k in r:
        #take batchsize slice of training data as matrix and perform training cycle on it
        #solutions have to be vectorized before stacking into matrix
        batch_input = train[:,k:k+network.batch]
        batch_solutions = vectorize_solutions(train_sol[k:k+network.batch])
        network.epoch(batch_input, batch_solutions, st)
    #after running through full training set, test on test data to see how network is progressing
    test_input = test[:,0:testsize]
    test_solutions = test_data[1][0:testsize]
    percent = network.test_network(test_input, test_solutions, testsize)



######SAVE IF IT DID WELL##########
print('Press "s" to save weights and biases, or any key to quit')
num = input("Enter: ")
if num == 's':
    network.save_weights(percent)
#######OTHERWISE, DESTROY IT AS AN EXAMPLE TO THE OTHERS#######



    





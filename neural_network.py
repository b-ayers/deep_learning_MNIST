import numpy as np
import gzip
import _pickle as cPickle


#for stochastic deep learning
#processes an entire batch at once as a matrix of batchsize columns
#rather than running each input through individually
class NeuralNet:
    #input list with number of nodes per layer, nlayer
    def __init__(self, nlayer, batchsize):
        self.L = len(nlayer)
        self.weights = []
        self.bias = []
        self.layers = nlayer
        self.batch = batchsize
        
        #for each layer after input layer generate weight matrix and bias
        for i in range(1,self.L):
            #each weight matrix should have dimension n_i x n_(i-1)
            self.weights.append(np.random.triangular(-0.5,0,0.5,(nlayer[i],nlayer[i-1])))
            #generates (n_i x batchsize) matrix of a repeated randomized column vector
            #because feeding entire batch through at once, may be easier to just add the
            #bias as a matrix rather than adding the same column vector to each column in z
            #self.bias.append(np.full((batchsize,nlayer[i]), np.random.rand(1,nlayer[i])).T)
            self.bias.append(np.random.triangular(-0.5,0,0.5,(nlayer[i],1)))
                                            
    #send in batch as a n_0 by N matrix input and n_L by N solution matrix
    def epoch(self, batch_in, batch_sol, step):
        a = [batch_in]
        z = []
        #calculate z and the sigmoid of z for each layer
        for i in range(0,self.L-1):
            #a starts from first layer, while w and b start from second layer,
            #so (z = w_l*a_(l-1) + b_l) works with same index
            z.append(np.dot(self.weights[i],a[i])+np.tile(self.bias[i], self.batch))
            a.append(sigmoid(z[i]))
        
        #list of dl, errors, starting with layer L, will add others to front
        dl = [(a[-1]-batch_sol) * sigmoidprime(z[-1])]
        
        #range of final z index to z[1], stepping backwards
        for i in range(len(z)-1, 0, -1):
            #stepping backwards, push d_l = (w_(l+1))^T*d_(l+1) product f'(z_l)
            dl.insert(0, np.dot(self.weights[i].T, dl[0])*sigmoidprime(z[i-1]))

        #using list of dl, for each layercalculate weight gradient matrix and basis gradient vector

        #for weight gradient matrices, dC_x/dw_jk^l = d_(l,x) * (a_(l-1,x))^T
        #where x indicates the xth column of d_l and a_(l-1)
        #the matrix product of these column/transposed row pairs is the weight gradient matrix
        #contribution from the xth training sample
        #adding them up and taking the average, (then normalizing(?)) gives total weight grad
        #repeat for each layer
        
        dwl = []
        for i in range(0,len(dl)-1):
            temp = np.zeros(np.shape(self.weights[i]))
            for x in range(0, self.batch):
                #a has same index because it starts on input layer
                temp += np.dot(dl[i][:,x:x+1], a[i][:,x:x+1].T)
            #average each over batch size
            temp = temp / self.batch
            dwl.append(temp)
        
        #for b gradient vector
        dbl = []
        for l in range(0,len(dl)-1):
            temp = dl[l][:,0:1]
            for x in range(1, self.batch):
                temp += dl[l][:,i:i+1]

            #average
            temp = temp / self.batch
            dbl.append(temp)

        #update weights
        for i in range(0, len(self.weights)-1):
            self.weights[i] = self.weights[i] - step*dwl[i]
        
        #update biases
        for i in range(0, len(self.bias)-1):
            self.bias[i] = self.bias[i] - step*dbl[i]



    #using test data separate from training sets
    #here the batch_sol is an array of the correct numbers, 0 through 9, as integers
    #rather than each being a column vector with a 1 in the appropriate spot
    def test_network(self, batch_in, batch_sol, size):        
        a = [batch_in]
        for i in range(0,self.L-1):
            a.append(sigmoid(np.dot(self.weights[i],a[i])+np.tile(self.bias[i], size)))
            
        #find index of max value in each output vector (column of a_L matrix)     
        output = np.argmax(a[-1], 0)

        correct = 0
        for x, y in zip(output, batch_sol):
            if x == y:
                correct += 1
        print(correct, ' out of ', size, 'evaluated correctly: ', (correct/size)*100, '%')

    def save_weights(self, step):
        fn = 'save_'
        for i in self.layers:
            fn = fn + str(i) + '_'
        fn += ('step_' + str(step) + '_batch_' + str(self.batch))
        fnw = fn + '_weights'
        with open(fnw, 'wb') as outfile:
            cPickle.dump(self.weights, outfile)
        fnb = fn + '_biases'
        with open(fnb, 'wb') as outfile:
            cPickle.dump(self.bias, outfile)

    def load_weights(self, weight_filename, bias_filename):
        with open(weight_filename, 'rb') as infile:
            self.weights = pickle.load(infile)
        with open(bias_filename, 'rb') as infile:
            self.bias = pickle.load(infile)


#compute the sigmoid function of each entry in matrix z and return as a matrix    
def sigmoid(z):
    return 1/(1+np.exp(-z))

#derivative of the sigmoid function
def sigmoidprime(z):
    return sigmoid(z)*(1-sigmoid(z))
        
#pass in an array of solutions and output a matrix where each column is
#the corresponding vectorized number        
def vectorize_solutions(arr):
    temp = np.zeros((10,arr.size))
    for i in range(0,arr.size):
        temp[arr[i],i] = 1
    return temp




#code to read in MINST data and format into batch size matrices
#data pulled from Neilson's file, going to reformat it myself
f = gzip.open('data/mnist.pkl.gz', 'rb')
imported = cPickle.Unpickler(file=f, encoding='latin1')
training_data, validation_data, test_data = imported.load()
f.close()

#training_data is a tuple with the first entry being a numpy array with 50,000 entries,
#each of which is an array of 784 values for each 28x28 image
#looks like training_data[0][1] for example returns the first entry in the tuple,
#which is the array of 50,000 images, and then the second image in that array, which is
#a long 1D row of all 784 entries, so I should just take that row and make it a column

train = training_data[0].T
test = test_data[0].T
train_sol = training_data[1]


#choosing different step size, batch size, and layer structure has widely varying results
#as well as effects of random weight initialization, some hands are better than others
#need to learn more about optimizing these parameters
#MAKE SURE for now that 5000 is evenly divisible by chosen batch size
network = NeuralNet([784, 200, 100, 50, 10], 10)

#how many times to cycle through the full training set
cycles = 40
#step size, can influence results dramatically, needs to be adjusted when changing other parameters
st = 10.0
testsize = 10000

for c in range(0, cycles):
    #make sure for now that 50000 (training set size) is EVENLY DIVISIBLE by chosen batch size,
    #this will generate a random permutation of batch size separated starting points
    r = np.random.permutation(range(0, 50000, network.batch))

    for k in r:
        batch_input = train[:,k:k+network.batch]
        batch_solutions = vectorize_solutions(train_sol[k:k+network.batch])
        network.epoch(batch_input, batch_solutions, st)
        

    test_input = test[:,0:testsize]
    test_solutions = test_data[1][0:testsize]

    network.test_network(test_input, test_solutions, testsize)

print('Press "s" to save weights and biases, or any key to quit')
num = input("Enter: ")
if num == 's':
    network.save_weights(st)
    
        
    
    




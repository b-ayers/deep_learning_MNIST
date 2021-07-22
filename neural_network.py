import numpy as np
import gzip
import _pickle as cPickle


#for stochastic deep learning
#processes an entire batch at once as a matrix of batchsize columns
class Neural_Network:
    #input list with number of nodes per layer, nlayer
    def __init__(self, nlayer, batchsize):
        self.L = len(nlayer)
        self.layers = nlayer
        self.batch = batchsize

        self.weights = []
        self.bias = []
        #for each layer after input layer generate weight matrix and bias
        for i in range(1,self.L):
            #each weight matrix should have dimension n_i x n_(i-1)
            self.weights.append(np.random.triangular(-0.5,0,0.5,(nlayer[i],nlayer[i-1])))
            #store bias as column vector for each layer, tile it later
            self.bias.append(np.random.triangular(-0.5,0,0.5,(nlayer[i],1)))


                                            
    #send in batch as a n_0 by N matrix input and n_L by N solution matrix
    #performs training cycle on single batch
    def epoch(self, batch_in, batch_sol, step):
        #set 784*N training set matrix as first layer node values
        a = [batch_in]
        z = []
        #calculate z and the sigmoid of z for each layer
        for i in range(0,self.L-1):
            #a starts from first layer, while w and b start from second layer,
            #so (z = w_l*a_(l-1) + b_l) works with same index
            z.append(np.dot(self.weights[i],a[i])+np.tile(self.bias[i], self.batch))
            a.append(sigmoid(z[i]))
        
        #list of dl, errors, starting with layer L, will add others to front
        #this is (dC/da_L)(da_L/dz_L)
        dl = [(a[-1]-batch_sol) * sigmoidprime(z[-1])]
        
        #range of final z index to z[1], stepping backwards
        for i in range(len(z)-1, 0, -1):
            #stepping backwards, push d_l = (w_(l+1))^T*d_(l+1) product f'(z_l)
            #basically, build the chain rule up step by step
            dl.insert(0, np.dot(self.weights[i].T, dl[0])*sigmoidprime(z[i-1]))

        #using list of dl, for each layer calculate weight gradient matrix and basis gradient vector

        ##for weight gradient matrices, dC_x/dw_jk^l = d_(l,x) * (a_(l-1,x))^T
        ##where x indicates the xth column of d_l and a_(l-1)
        ##the matrix product of these column/transposed row pairs is the
        ##weight gradient matrix contribution from the xth training sample
        ##adding them up and taking the average, gives total weight grad
        ##repeat for each layer
        
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
        return (correct/size)*100



    def save_weights(self, percent):
        
        fn = 'saves\p' + str(percent) + 'save_'
        for i in self.layers:
            fn = fn + str(i) + '_'

        fnw = fn + 'weights.pkl'
        with open( fnw, 'wb') as outfile:
            cPickle.dump(self.weights, outfile)

        fnb = fn + 'biases.pkl'
        with open(fnb, 'wb') as outfile:
            cPickle.dump(self.bias, outfile)



    def load_weights(self, weight_filename, bias_filename):
        with open(weight_filename, 'rb') as infile:
            self.weights = cPickle.load(infile)
        with open(bias_filename, 'rb') as infile:
            self.bias = cPickle.load(infile)


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


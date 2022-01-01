# deep_learning_MNIST

*Note from future: this was my first step into machine learning. It is a first principles
implementation of multi-layer FC neural networks with stochastic gradient descent. 
It was intended for educational purposes only: for understanding the nuts and bolts
of how NNs produce results that otherwise look like magic. Any serious application of
neural networks should use a framework like torch or keras, which have been optimized
for performance and modularity. (The 98.5% accuracy I achieved here in a few days of work
took only a few hours of pytorch time (including learning the basic library).) In this 
readme I attempted to explain my understanding in relatively simple terms for anyone 
who does't have a strong calculus foundation.*

This is my implementation of a neural network with stochastic gradient descent,
using principles learned in Michael Nielsen's book found at the link:
http://neuralnetworksanddeeplearning.com/chap1.html
Additionally, an excellent visual tour can be found on 3blue1brown's youtube channel.
The main difference and (hopefully) improvement over Nielsen's example implementation 
is that instead of feeding through one training sample at a time and then totaling the
results of a batch, an entire batch is fed through at once, where each individual
training sample is the column of a matrix. Check out Nielsen's book for a more detailed
and excellent description, but I will write a summary here, including the differences
in my implementation.

The network consists of an arbitrary number of layers with an arbitrary number of 
nodes in each layer. The jth node in the lth layer recieves input from every node
in the (l-1)th layer. That input is the output value of the respective node, multiplied
by a weight w_j,k. For example, suppose layer l has 3 nodes and layer (l-1) has 4 nodes,
then the first node in layer l recieves input which is the sum total of the weighted 
outputs from the four nodes in (l-1), or

z_1,l = w_1,1 * x_1,(l-1) + w_1,2 * x_2(l-1) + w_1,3 * x_3,(l-1) + w_1,4 * x_4,(l-1)

where z_1,l is the raw input to node 1 in layer l, and x_k,(l-1) represents the raw
output from the kth node in the (l-1)th layer. Subscripts are clunky in plain text.
If we play around with it, we can notice that this type of multiplication can be 
put into matrix form if we make a matrix of weights associated with the lth layer
of dimension n_l x n_(l-1) where n is the number of nodes in the layer. So the matrix 
in this example will look something like

[w_11   w_12    w_13    w_14]

[w_21   w_22    w_23    w_24]

[w_31   w_32    w_33    w_34]

and if we multiply that matrix on the right side by the column vector

[x_1,(l-1)]

[x_2,(l-1)]

[x_3,(l-1)]

[x_4,(l-1)]

we get a column vector in which each entry is the raw input

[z_1,l]

[z_2,l]

[z_3,l]

However, we can also realize that, instead of multiplying the weight matrix by a single
column vector of inputs, we could instead stack a bunch of column vector inputs together
to form a (in this case) 4xN matrix, where N is the number of training samples we want
to feed through the network. Then the column of raw inputs z will also become a 3xN matrix
where the ith column is the result for the ith training sample. (Note that the weight matrix
size still only depends on the size of the two layers involved.)

In fact, however, I've left out one parameter so far. In addition to the weighted raw input
from the previous layer of nodes, we add to each raw input a bias as an additional parameter, 
so my previous equation in fact should have been:

z_1,l = w_1,1*x_1,(l-1) + w_1,2*x_2(l-1) + w_1,3*x_3,(l-1) + w_1,4*x_4,(l-1) + b

If we were feeding through a single training sample, this would amount to adding the column vector

[b_1,l]

[b_2,l]

[b_3,l]

to the raw input I've already mentioned. Since we're feeding through a batch of size N, however,
we can think of it as tiling the above column vector N times into a 3xN matrix, (call it B) and
then adding that to the input.

The total matrix equation then (using capital letters to represent matrices and vectors), 
should be

Z_l = (W_l)(X_(l-1)) + B

where each entry in the column vector Z_1 is the raw intput z_j to the jth node in the lth layer.
Depending on the weights and biases involved, this raw input could be any number. A function is
then applied to kind of rein in this wild distribution. Neilsen first teaches the classic sigmoid,
or logisitic function, 1/(1+e^(-z)) which is what is being used here at time of writing. For 
large positive raw inputs, z, the logistic function asymptotes to 1, while for large negative 
raw inputs it asymptotes to 0, thus constraining the final value of each node to fall 
between  0 and 1. This process continues feeding forward, with a different set of weights and biases
associated with each layer in the network, until we finally assign values to the last layer of nodes.
The output of the final layer is interpreted based on the problem we're trying to solve. 
Here we are trying to read handwritten numbers from the MNIST database, so we have 10 nodes in the 
final layer. Ideally, once the system is trained, it spits out a 1 in the space of the correct
digit, 0 through 9. Of course, the system should never actually become completely certain, just
like we're not always certain when reading someone else's handwriting, so what we actually do is
go through and find the node with the largest value and interpret that as the answer. 

Of course, since we start with the weights and biases initialized to a random distribution of values,
the initial responses of our network will be no better than random garbage. In this case, it should
give the correct answer about 10% of the time. Only by adjusting the weights and biases can we nudge
the network toward the best possible version of itself. How do we know which way to turn the dials
to do this? We have a cost function that we run on the output of the last layer. This cost function
serves as a heuristic to tell us how much the network just sucked at what it was supposed to be doing.
At time of writing, the cost function implemented here is the quadratic C=(1/2)(X_L-Y)^2, where X_L is 
the column vector of outputs for the final layer of nodes and Y is the column vector of what they SHOULD
be, which is the solution that was paired with the training sample that got fed in on the first layer.** 

**(I've failed to mention so far that the first layer of the network in this case contains 784 nodes,
each of which represents a grayscale value of a pixel in a 28x28 image of a handwritten digit, pulled
from MNIST's database. So the node structure of our layers for this example always starts with a layer
of 784 nodes and ends with a layer of 10 nodes.)

The interpretation is straightforward: the larger the difference between the system's output X_L and 
the correct answer Y, the larger the cost function will be. Our ideal final network minimizes C.

I'm not sure why the 1/2 is there in the cost function, other than to make the derivative work out nicely,
and the derivitive is what we will actually calculate, rather than the cost function itself. 
Note: I'm likely to change the cost functionand sigmoid function as I learn more techniques.

So the cost function is designed to tell us how wrong we are. Still, knowing how wrong something is
doesn't immediately tell us what to change to fix it. Rather, we want to know how the cost function
changes when we twiddle the dials of our weights and biases. That is, we want to know the rate of
change of the cost function with respect to a change in a specific weight or bias. This is a derivative:
dC/dw (or dC/db for a bias).

Consider the weights of the output layer, L. Let's make it concrete: suppose layer L-1 has only 3 nodes, 
so the three weights influencing the value of the first node in layer L (the node corresponding to 
the output interpretation of a 0) are w_11, w_12, and w_13, and the weight matrix looks like this:
(I'm dropping layer subscripts for now)

[w_11   w_12    w_13]

[w_21   w_22    w_23]

[w_31   w_32    w_33]

[w_41   w_42    w_43]

[w_51   w_52    w_53]

[w_61   w_62    w_63]

[w_71   w_72    w_73]

[w_81   w_82    w_83]

[w_91   w_92    w_93]

[w_10,1   w_10,2    w_10,3]

Suppose we want to know how C changes when we change w_11 a little bit. Well, we have to consider
the cascading chain of what affects what when we change w_11 a little. w_11 is used in calculating
the value of the raw input to the first node of layer L as:

z_1,L = w_11 * x_1,L-1 + w_12 * x_2,L-1 + w_13 * x_3,L-1 + b

then z_1 is used to calculate the final value of that node using

x_1,L = sigmoid(z_1)

then the cost function is calculated for that node as

C_1 = (1/2)(x_1,L-y_1)^2

So changing w_11 changes z_1 a little bit, which changes x_1 a little bit, which changes the cost
function a little bit. By how much? Well, consider the first step, where we calculate z_1 from w_11. 
How much does z_1 change if we change w_11 by an amount dw_11? If we knew how much z_1 changed, dz_1,
per change in w_11, then we could represent the change with fractions by (dz_1/dw_11) * (dw_11) = dz_1. 
And for the next step, if we knew how much x_1 changed (dx_1) for a small change in z_1 (dz_1), then
we could represent that change by (dx_1/dz_1) * (dz_1) = dx_1, or we could plug in our symbolic expression
for dz_1 and get dx_1 = (dx_1/dz_1) * (dz_1/dw_11) * (dw_11), the change in dx_1 per small change dw_11.
Taking the final step, the change in C_1, dC_1, per small change in x_1 is (dC_1/dx_1) * (dx_1) = dC_1,
but we know dx_1 in terms of dw_11, so we can put it all together as:

dC_1 = (dC_1/dx_1) * (dx_1/dz_1) * (dz_1/dw_11) * dw_11

which is how much the cost function of the first output node changes when we change dw_11. This has just
been a description of the chain rule of calculus. To ask how C changes with respect to changes in w, it 
asks, "well, how does C change when we change x" and then "yes, but then how does x change when we change z"
and "yes, but then how does z change when we change w", and then it multiplies all those rates together
to find the rate at which the cost function changes with respect to that weight:

dC_1/dw_11 = (dC_1/dx_1) * (dx_1/dz_1) * (dz_1/dw_11)

This rate is like the slope of a hill. If I am on a hill, and I want to get to the bottom of the valley
(i.e. where the cost function is minimized), then I want my steps to be in directions of maximum descent,
or maximum slope. Thus, supposing I have a fixed step size, then a larger dC_1/dw_11 will take me to the
valley faster. So what we do is basically calculate this slope in every direction in our multidimensional space
and take larger steps in the directions that get us the most change. Effectively, we perform these calculations
and compile the matrix of slopes:

[dC/dw_11   dC/dw_12    dC/dw_13]

[dC/dw_21   dC/dw_22    dC/dw_23]

...

...

...

...

...

...

[dC/dw_91   dC/dw_92    dC/dw_93]

[dC/dw_10,1   dC/dw_10,2    dC/dw_10,3]

Call this matrix D. Then we multiply this entire matrix by a fixed step size. The entries with larger slopes 
will then contain larger numbers, and we will subtract this matrix from our matrix of weights, so that we update
it as

W = W - stepsize * D

We do this for every weight matrix W and then run the whole process again, and update again.
This is how we get down the hill and minimize the cost function by changing the weights that get us there 
most effectively. You can imagine that if a weight is already the best possible version of itself, then
changing it will not change C, and so that entry dC/dw = 0 and we will not change that weight when we update W.
The same idea applies to changing the biases by calculating dC/db. 

All this to try to understand the basic principle. I've been tiptoeing with respect to calculus, but 
to calculate the derivatives (i.e. dC/dx, dx/dz, dz/dw), we'll need to actually apply what is learned in a 
first calculus course, so I will assert rather than really explain. 

Continuing with the concrete example, looking at the derivatives involved with w_11

If C_1 has the quadratic form (1/2)(x_1-y_1)^2 then the derivative is just (x_1-y_1). 

Now x_1 = sigmoid(z) - y_1 = 1/(1+e^(-z_1)) - y_1 and the derivative of that is 

dz_1 = e^(-z_1)/(1+e^(-z_1))^2,

or, more conveniently for the program, 

sigmoid(z_1)*(1-sigmoid(z_1)). 

Now, for this concrete example, 

z_1,L = w_11 * x_1,L-1 + w_12 * x_2,L-1 + w_13 * x_3,L-1 + b

and dz_1/dw_11 = x_1,L-1

And we string those three derivatives together to get dC_1/dw_11. I've been somewhat messy with my
subscripts because they clutter things up in text format.
The question becomes how to do all of this most efficiently, since as we go backwards into our
network to see what effects what at what rate, everything compounds. For example, if we go one layer
further back, and ask what effect a weight from that layer has on C, well, that weight changed the input
to one node on layer L-1, and then that single node fed input into every output node in layer L, and
we have to add up all those effects. The only thing that saves the situation for us, is that property
of the chain rule where we kept stacking the rates of change on top of each other and multiplying them
together. So every time we go back a layer we keep the derivatives calculated in the previous step
and tack some new ones on. If I go too far into this process of backpropagation I may find myself
writing a book myself, and that has already been done, so I point you to Nielsen's second chapter. 

The only difference from Nielsen's implementation with my backpropagation is that his dl's and a_is 
are column vectors and mine are matrices, where each column is associated with one of the training
samples being fed through simultaneously. So where he does

dl_l * a.transpose_(l-1) to get the gradient matrix (what I previously called D), I do

            temp = np.zeros(np.shape(self.weights[i]))
            for x in range(0, self.batch):
                #a has same index because it starts on input layer
                temp += np.dot(dl[i][:,x:x+1], a[i][:,x:x+1].T)
            #average each over batch size
            temp = temp / self.batch


Basically, I calculate a D_i for each training sample using each column of dl and a, then add them 
all together and average to get my final D which I use to update W by

W = W - stepsize * D

A similar difference applies to updating the biases.


As for some other details of the program,

--changing batchsize changes how many training samples are fed through at once (how many are columns
are stacked together into a matrix), for now it should just be something that the full training set is
divisible by since I haven't added any exceptions for handling otherwise

--changing step size can lead to widely varying results. For example, suppose you're a giant trying to
get to the bottom of the valley (cost function minimum) but you can only take steps the size of a 
football field, like your mother taught you. Then at every step you orient yourself in the direction
of the valley minimum, but you always overstep it, and you turn around, and you overstep it again, etc.
or an especially small step size might not get you anywhere at all. There's also questions of getting 
stuck in local minima vs. global minima, which I need to learn more about. 

--there are better cost functions and "sigmoid" functions to implement

--changing "cycles" changes how many times you loop through the training set. At some point you tend
to asymptote based on the network parameters you've chosen, and there's no point in continuing on
without changing parameters. The best I have achieved so far as of writing this is 98.5% of the 10,000
testing set correct. I think the best networks get around 99.8%(?), so plenty of room for improvement
in technique and parameters

--there are rudimentary save and load functions. If your network has been a good network, hit 's' to save
the weights and biases. If not... well... we don't hear from those networks again. 


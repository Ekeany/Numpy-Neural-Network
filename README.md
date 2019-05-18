# Numpy-Neural-Network

## Introduction.
Artificial neural networks are a form of supervised machine learning algorithm. They were first
developed in the 1950s to test hypotheses about the interaction of biological neurons. The human
brain consists of a complex signal network of nerve cells or neurons each connected in layers. These
neurons communicate in parallel meaning that each neuron receives input lines from all the neurons
in the previous layers if these signals are above a certain threshold the neuron will be stimulated and
emit its own signal to all the neurons in the next layer.

The artificial neural network is a mathematical algorithm that consists of three main sections an
input layer, hidden layer(s) and an output layer. These layers contain numerous sets of
interconnected artificial neurons. These neurons are essentially just a node where the main
computation of the model occurs. A neuron combines input data from multiple neurons in the
previous layer with a set of weighted coefficients that either amplify or reduce the response of that
individual signal, thus assigning significance to each previous neuron in the input. These weighted
inputs are then summed together and passed through an activation function in order create its own
output which will then be permeated to the next layer.

The ability to learn is the driving concept behind the neural network. To achieve this, the network
seeks to find the optimal parameters to solve the given classification task. Before the training
process can begin the parameters of the network have to be initialized these initial values are often
chosen at random, however faster convergence with the optimal values can be achieved by
heuristically assigning the weights. The learning process commences by feeding the training data
through the network. This training data consists of input-output pairs with the outputs representing
the “labels” or “targets”. First the inputs are fed into the network, then the activations of all the
nodes in the hidden layer are calculated and the activation of the output layer is collected, this
sequence of events is referred to as “Feed Forward”. As the learning process is iterative, the output
produced from the feed forward for each input in the training set is analyzed and the network
parameters are repeatedly adjusted in accordance. This training process continues until a
satisfactory level is achieved, (MSE) was the metric chosen to assess the networks performance.

The most common learning method implemented by neural networks is Backpropagation. This
algorithm calculates the gradient of the error function with respect to the neural networks weighted
coefficients, it then uses gradient descent to optimize these weighted coefficients. The gradient is
permeated backwards through the entire network, with the gradient of the output layer weights are
calculated first and then the hidden layer. Partial computations of the gradient from one layer are
recycled in the calculations of the gradient for the previous layer. This is a supervised process, as the
gradient of the error is achieved by comparing the output of the network with the target label. The
computed global minimum represents the theoretical solution with the lowest possible error
therefore the weighted connections of each node are adjusted in correspondence to the direction of
this gradient. This process is repeated for each epoch or iteration until the model has achieved a
satisfactory level.

## Methods.
Two matrices were created from the csv file the first matrix contains the 4 input feature columns
and corresponding labels were stored in a separate matrix. A preprocessing step was needed to
convert the labels into a format called one hot encoding. As the output layer produces a number of
activated outputs a threshold is applied producing a sequence of zeros and ones. Therefore the
labels have to be converted to a similar format for our network to train. 

<p align="center">
  <img width="533" height="34" src="/Images_/image1.PNG">
</p>

The network is then established by initializing matrices to represent the weights and biases found
between each layer of the network, for this project a simple neural network was established with
only hidden layer. The dimensions of these weight matrices are predetermined by the amount of
features and labels in the dataset along with the number of nodes chosen in the hidden layer. The
weights and biases are heuristically chosen to range from positive – negative one as the sigmoid
activation function will transform any value to this range anyways.

<p align="center">
  <img width="567" height="109" src="/Images_/image2.PNG">
</p>

From the code above there are four matrices representing the weights and biases for the input to
hidden and the hidden to output layers.The bias units aren't connected to any previous layer and in
essence can be compared to the intercept of the line equation as it is able to shift the activation
function making the neuron output more flexible. An activation function simple maps a value into
binary or into a binary range mimicking the biological neuron as if a threshold is applied then the
neuron is either firing or not.

<p align="center">
  <img width="674" height="237" src="/Images_/image3.PNG">
</p>

To begin the feedforward section of the algorithm the input vector is multiplied by the matrix dot
product of the randomly initialized weights for the input to hidden layer the biases are then added
and a nonlinear transformation using an activation function is applied. The sigmoid function was
chosen for this project, however there are numerous other activation functions. The sigmoid
function provides a relatively easy derivative and an un-normalized probability output which
simplifies classification tasks. The activation from the hidden layer then goes through the same
process producing the output activations.

<p align="center">
  <img width="656" height="97" src="/Images_/image4.PNG">
</p>

Backpropagation begins by calculating the error at the output layer after the first feedforward
iteration. This error is defined as the difference between the activated output and the target values
multiplied by the derivative of the sigmoid function at that point.

<p align="center">
  <img width="389" height="59" src="/Images_/image5.PNG">
</p>

From the output error using the chain rule the error can be permeated backwards and the error for
each neuron in the hidden layer can be calculated. The error for a neuron in the hidden layer is
defined as the sum of the products between the errors of the neurons in the output layer and the
weights associated with the neurons in that layer, multiplied by the derivative of the sigmoid
function. 

<p align="center">
  <img width="476" height="54" src="/Images_/image6.PNG">
</p>

These errors are then used to find the variation of the weights based on the current input and
output patterns. The variation or delta is the negative gradient of the cost function. It can be
expressed as the product of the current weighted coefficient between two nodes in two separate
layers with the error associated at that node in the further most right layer and the learning rate.
The learning rate scales the magnitude of the variation helping the function to slowly settle at the
global minimum instead of overshooting. As the biases are note actively connected they are updated
network can be updated from the aggregated weighted errors at that specific neuron.

<p align="center">
  <img width="364" height="99" src="/Images_/image7.PNG">
</p>

<p align="center">
  <img width="663" height="190" src="/Images_/image8.PNG">
</p>

## Results.

<p align="center">
  <img width="631" height="401" src="/Images_/image9.PNG">
</p>

Figure 3.1: This graph depicts the mean square error of the training set as a function of epochs for a
network of 4 input nodes 66 hidden nodes and 3 output nodes. As expected the error converges to
the global minimum as the number of iterations tends to infinity. The error ranges from 0.25 to 0.07,
steadily decreasing as the number of iterations increases. For the duration of the project an epoch of
80000 was chosen as it is the first point to constantly have an MSE below 0.1 thus saving training
time for the model.

<p align="center">
  <img width="632" height="411" src="/Images_/image10.PNG">
</p>

Figure 3.2: This plot describes the average learning rate for 10 random divisions. The graph was
produced by training the model on a varying number of random elements chosen from the training
set, the accuracy at each sample size of the training set was computed by calculating the
corresponding confusion matrix and the formula below. The average accuracy over all three classes
was then taken and plotted. 

<p align="center">
  <img width="267" height="88" src="/Images_/image11.PNG">
</p>

It is evident from the graph that the accuracy converges to one as the number of training examples
increases. Thus the classifier transitions from a high bias to a low bias. The jagged nature especially
at lower sample sizes indicates a high variance within the model. Therefore the model requires at
least 70 training instances to accurately and consistently predict the validation set. 

<p align="center">
  <img width="570" height="492" src="/Images_/image12.PNG">
</p>

<p align="center">
  <img width="719" height="161" src="/Images_/image13.PNG">
</p>

A confusion matrix was created from the 10 random training and testing splits. The diagonal cells
refer to the number of correctly classified cases and the misclassified cases are shown by the nondiagonal cells. From the average accuracy, the network classified 98% of the testing cases correctly
and 2% incorrectly. This 2% inaccuracy could be attributed to the misclassification of the “Barn Owl”
class which had the most inconsistent predictions as it has been repeatedly mistaken for the “Long
Eared Owl”.

## Conclusion
The Neural Network algorithm provides an excellent approximation of the given supervised learning
task with 98% of cases being correctly predicted. Despite Neural networks being resource intensive,
as a whole they have little restrictions on what they can do. However, neural networks are
essentially black boxes which disconnect user and machine. Therefore their impressive capabilities
should complement, rather than substitute, human expertise and critical thinking.

## References.

1. Gupta S. 2017. hackernoon.com [Internet]. writing-a-neural-network-from-scratch-theory. Available
at: https://hackernoon.com/dl01-writing-a-neural-network-from-scratch-theory-c02ccc897864
2. Varzaru M. 2010. Mihai’s Weblog [Internet]. Backpropagation algorithm. Available at:
https://mihaiv.wordpress.com/2010/02/08/backpropagation-algorithm/
3. Bataineh H M. 2012. Iowa Research online [Internet]. Artificial neural network for studying human
performance. Available at: https://ir.uiowa.edu/cgi/viewcontent.cgi?article=3260&context=etd
4. Nielsen A M. 2015. Determination Press [Internet]. Neural Networks and Deep Learning. Available
at: http://neuralnetworksanddeeplearning.com/chap2.html
5. Usman M. 2018. Stackabuse.com [Internet]. creating a neural network from scratch in python.
Available: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

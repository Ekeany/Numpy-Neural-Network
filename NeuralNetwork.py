class NeuralNetwork():
    
    def __init__(self, inputlayer_neurons, hiddenlayer_neurons, output_neurons):

        #Set training iterations
        # and learning rate
        self.epoch = 8000
        self.lr = 0.001 

        #weight and bias initialization
        self.weights_ih = np.random.uniform(-1,1,size=(inputlayer_neurons,hiddenlayer_neurons))
        self.bias_ih = np.random.uniform(-1,1,size=(1,hiddenlayer_neurons))
        self.weigths_ho = np.random.uniform(-1,1,size=(hiddenlayer_neurons,output_neurons))
        self.bias_ho = np.random.uniform(-1,1,size=(1,output_neurons))
    

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def derivatives_sigmoid(x):
        return x*(1 - x)

    @staticmethod
    def MeanSquaredError(y, y_pred):
        return(np.mean((y - y_pred)**2))


    def FeedForward(self,X):
        #Forward Propogation       
        self.hiddenlayer_activations = self.sigmoid(np.dot(X,self.weights_ih)+ self.bias_ih)
        output_layer = np.dot(self.hiddenlayer_activations,self.weigths_ho)+self.bias_ho
        return(self.sigmoid(output_layer))


    def Fit(self,X,y):
        self.y = y
        for i in range(self.epoch):
            # Forward Propagation
            output = self.FeedForward(X)
            
            #Backpropagation
            E = self.y-output
            
            d_output = E * self.derivatives_sigmoid(output)
            d_hiddenlayer = d_output.dot(self.weigths_ho.T) * self.derivatives_sigmoid(self.hiddenlayer_activations)

            self.weigths_ho += self.hiddenlayer_activations.T.dot(d_output) *self.lr
            self.bias_ho += np.sum(d_output, axis=0,keepdims=True) *self.lr
            self.weights_ih += X.T.dot(d_hiddenlayer) *self.lr
            self.bias_ih += np.sum(d_hiddenlayer, axis=0,keepdims=True) *self.lr
            if i % 1000 == 0:
                print("Loss: \n" + str(i) +"  " +str(np.mean(np.square(self.y - output))))
                 
    def Predict(self,Input):
      return(self.FeedForward(Input))

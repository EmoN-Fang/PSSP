import math
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

import time
time_start=time.time()

X_train = np.load("X_train.npy")
Y3_train = np.load("Y3_train.npy")
Y8_train = np.load("Y8_train.npy")
X_test = np.load("X_test.npy")
Y3_test = np.load("Y3_test.npy")
Y8_test = np.load("Y8_test.npy")

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x,None]) 
    Y = tf.placeholder(tf.float32, [n_y,None])
    
    return X, Y

def initialize_parameters():

    
    tf.set_random_seed(3)                   # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [10,20], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b1 = tf.get_variable("b1", [10,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [5,10], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b2 = tf.get_variable("b2", [5,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [3,5], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
    b3 = tf.get_variable("b3", [3,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)                                          # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                               # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                                # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                               # Z3 = np.dot(W3,Z2) + b3

    
    return Z3

def compute_cost(Z3, Y):
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    np.random.seed(seed)           
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((3,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1)* mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1)* mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def model(X_train, Y_train, X_test, Y_test, learning_rate0 = 0.01,
          num_epochs = 10, minibatch_size = 1024, print_cost = True):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(0)                             # to keep consistent results
    seed = 0                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
#     current_iter = tf.Variable(0, trainable=False)
#     learning_rate = tf.train.exponential_decay(learning_rate0, current_iter, 3500, 0.9, staircase=False)  
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate0).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
#         # plot the cost
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate0))
#         plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

parameters = model(X_train, Y3_train, X_test, Y3_test)
time_end=time.time()
print('totally cost',time_end-time_start)
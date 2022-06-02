import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)
      This adjusts the weights to minimize loss.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    def normalized(array):
        array_max = array.max()
        array_exp = np.exp(array - array_max)
        return array_exp / np.sum(array_exp)


    def normalized_vec(x):
        x_max = np.max(x)
        x_exp = np.exp(x - x_max)
        x_row_sum = np.sum(x_exp, axis=1, keepdims=True)
        f = x_exp / x_row_sum
        return f


    def extend(y):
        y_extended = np.zeros((y.shape[0], y.max() + 1))
        y_extended[np.arange(y.shape[0]), y] = 1
        return y_extended
    
   
    y_extended = extend(y)
    
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1] # 1
    
        
    for X_i, y_extended_i in zip(X, y_extended):
        f = normalized(np.dot(X_i, W))
        loss_i = np.dot(-y_extended_i, np.log(f))
        loss += loss_i
        for idx in range(W.shape[1]):
            dW[:, idx] += -X_i.transpose() * (y_extended_i[idx] - f[idx])

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + reg * 2 * W

    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    def normalized(array):
        array_max = array.max()
        array_exp = np.exp(array - array_max)
        return array_exp / np.sum(array_exp)


    def normalized_vec(x):
        x_max = np.max(x)
        x_exp = np.exp(x - x_max)
        x_row_sum = np.sum(x_exp, axis=1, keepdims=True)
        f = x_exp / x_row_sum
        return f


    def extend(y):
        y_extended = np.zeros((y.shape[0], y.max() + 1))
        y_extended[np.arange(y.shape[0]), y] = 1
        return y_extended

    f = normalized_vec(np.dot(X, W))
    y_extended = extend(y)
    N = X.shape[0]
    loss = np.sum(-y_extended * np.log(f)) / N + reg * np.sum(W * W)
    dW += -np.dot(X.transpose(), (y_extended - f)) / N + reg * 2 * W

    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW

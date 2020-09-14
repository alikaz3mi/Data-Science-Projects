# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:44:41 2020

@author: AliKazemi
"""
import numpy as np
import matplotlib.pyplot as plt


class ThreeLayerNet(object):
  """
  A Three-layer fully-connected neural network.

  """

  def __init__(self, input_size, hidden1_size, hidden2_size, output_size, std=1e-2):
      '''

      Initialize the model. Weights are initialized to small random values and
      biases are initialized to zero. Weights and biases are stored in the
      variable self.params, which is a dictionary with the following keys:

          W1: First layer weights; has shape (D, H1)
          b1: First layer biases; has shape (H1,)
          W2: Second layer weights; has shape (H1, H2)
          b2: Second layer biases; has shape (H2,)
          W3: Second layer weights; has shape (H2, C)
          b3: Second layer biases; has shape (C,)

      Parameters
      ----------
      input_size : TYPE
          DESCRIPTION. The dimension D of the input data.
      hidden1_size : TYPE
          DESCRIPTION. The number of neurons H1 in the 1st hidden layer.
      hidden2_size : TYPE
          DESCRIPTION. The number of neurons H2 in the 2nd hidden layer.
      output_size : TYPE
          DESCRIPTION. The number of classes C.
      std : TYPE, optional
          DESCRIPTION. The default is 1e-4.

      Returns
      -------
      None.

      '''
      self.params = {}
      self.params['W1'] = std * np.random.uniform(-1, 1, size = [input_size, hidden1_size])
      self.params['b1'] = np.zeros(hidden1_size)
      self.params['W2'] = std * np.random.uniform(-1, 1, size = [hidden1_size, hidden2_size])
      self.params['b2'] = np.zeros(hidden2_size)
      self.params['W3'] = std * np.random.uniform(-1, 1, size = [hidden2_size, output_size])
      self.params['b3'] = np.zeros(output_size)

  def loss(self, X, y, reg=0.0):
      '''
      Compute the loss and gradients for a three layer fully connected neural
      network.
      Inputs:
          - X: Input data of shape (N, D). Each X[i] is a training sample.
          - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
          - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.

      Parameters
      ----------
      X : TYPE
          DESCRIPTION.
      y : TYPE, optional
          DESCRIPTION. The default is None.
      reg : TYPE, optional
          DESCRIPTION. The default is 0.0.

      Returns
      -------
      TYPE
          DESCRIPTION.

      '''

      # Unpack variables from the params dictionary
      W1, b1 = self.params['W1'], self.params['b1']
      W2, b2 = self.params['W2'], self.params['b2']
      W3, b3 = self.params['W3'], self.params['b3']
      N, D = X.shape
      H1 = W1.shape[1]
      H2 = W2.shape[1]
      C = W3.shape[1]

      # Compute the forward pass
      scores = None

      # First Layer Pre-activation
      z1 = X.dot(W1) + b1

      # First Layer Activation
      a1 = np.maximum(0, z1)

      # Second Layer Pre-activation
      z2 = a1.dot(W2) + b2

      # Second Layer Activation
      a2 = np.maximum(0, z2)

      # Third Layer Pre-activation
      z3 = a2.dot(W3) +b3

      scores = z3

      # Compute the loss
      loss = None
      # In case of Saturation:
      #scores -= np.max(scores, axis=1, keepdims=True)

      if y.dtype=='int':
          exp_scores = np.exp(scores)
          a3 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
          correct_log_probs = -np.log(a3[range(N), y])
          data_loss = np.sum(correct_log_probs) / N

      else:
          data_loss = .5 * np.linalg.norm(scores-y) / N

      reg_loss = 0.5 * reg * (np.sum(W1 * W1) +np.sum(W2 * W2) + np.sum(W3 * W3))
      loss = data_loss + reg_loss



      # Backward pass: compute gradients
      grads = {}
      if y.dtype=='int':
          dscores = a3
          dscores[range(N), y] -=1
          dscores /= N
      else:
          dscores = scores - y[:, np.newaxis]
          dscores /= N

      grads['W3'] = np.dot(a2.T, dscores)
      grads['b3'] = np.sum(dscores, axis=0)

      dhidden2 = np.dot(dscores, W3.T)
      dhidden2[z2 <= 0] = 0

      grads['W2'] = np.dot(a1.T, dhidden2)
      grads['b2'] = np.sum(dhidden2, axis=0)

      dhidden1 = np.dot(dhidden2, W2.T)
      dhidden1[z1 <= 0] = 0

      grads['W1'] = np.dot(X.T, dhidden1)
      grads['b1'] = np.sum(dhidden1, axis=0)

      grads['W3'] += reg * W3
      grads['W2'] += reg * W2
      grads['W1'] += reg * W1

      return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
      '''

      Parameters
      ----------
      X : TYPE
          A numpy array of shape (N, D) giving training data.
      y : TYPE
          A numpy array f shape (N,) giving training labels.
      X_val : TYPE
          A numpy array of shape (N_val, D) giving validation data.
      y_val : TYPE
          A numpy array of shape (N_val,) giving validation labels.
      learning_rate : TYPE, optional
          DESCRIPTION. Scalar giving learning rate for optimization. The default is 1e-3.
      reg : TYPE, optional
          DESCRIPTION. The default is 1e-5.
      num_iters : TYPE, optional
          DESCRIPTION. Number of steps to take when optimizing. The default is 100.
      verbose : TYPE, optional
          DESCRIPTION. if true print progress during optimization. The default is False.

      Returns
      -------
      dict
          DESCRIPTION.

      '''

      num_train = X.shape[0]
      iterations_per_epoch = max(num_train / batch_size, 1)

      # Use SGD to optimize the parameters in self.model
      loss_history = []
      train_acc_history = []
      val_acc_history = []

      for it in range(num_iters):
          X_batch = None
          y_batch = None
          sample_indices = np.random.choice(num_train, batch_size)
          X_batch = X[sample_indices]
          y_batch = y[sample_indices]

          # Compute loss and gradients using the current minibatch
          loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
          loss_history.append(loss)
          self.params['W1'] += -learning_rate * grads['W1']
          self.params['b1'] += -learning_rate * grads['b1']
          self.params['W2'] += -learning_rate * grads['W2']
          self.params['b2'] += -learning_rate * grads['b2']
          self.params['W3'] += -learning_rate * grads['W3']
          self.params['b3'] += -learning_rate * grads['b3']

          if verbose and it % 100 == 0:
              print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

              # Every epoch, check train and val accuracy and decay learning rate.
          if it % iterations_per_epoch == 0:
              # Check accuracy
              train_acc = (self.predict(X_batch) == y_batch).mean()
              val_acc = (self.predict(X_val) == y_val).mean()
              train_acc_history.append(train_acc)
              val_acc_history.append(val_acc)

              # Decay learning rate
              learning_rate *= learning_rate_decay

      return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }

  def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
            - X: A numpy array of shape (N, D) giving N D-dimensional data points to
            classify.

        Returns:
            - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1)
        z2 = a1.dot(self.params['W2']) + self.params['b2']
        a2 = np.maximum(0, z2)
        scores = a2.dot(self.params['W3']) + self.params['b3']
        y_pred = np.argmax(scores, axis=1)
        return y_pred



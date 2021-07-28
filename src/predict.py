import pandas as pd
import numpy as np

# function uses the model and the initial input set to predict a given number of following values
def predict_values(model, initial_set, values, window_size):
    result = []
    for i in range(0, values):
        x_input = initial_set.reshape((1, window_size, 1))
        yhat = model.predict(x_input, verbose=0)
        result.append(yhat[0][0])
        
        # update the model input for the next prediction
        initial_set = np.append(initial_set, yhat)
        initial_set = np.delete(initial_set, 0)
        
    return result


# function uses the model and the initial input set to predict a given number of following values. 
# additionally the gradient is 
def predict_values_with_gradient(model, initial_set, initial, values, window_size):
    result = []
    for i in range(0, values):
        x_input = initial_set.reshape((1, window_size, 1))
        yhat = model.predict(x_input, verbose=0)
        result.append(yhat[0][0])
        
        # update the model input for the next prediction
        initial_set = np.append(initial_set, yhat)
        initial_set = np.delete(initial_set, 0)
        
        # calculate the gradient of the last values. If it falls under a threshold, 
        # the input data will be reseted to 140.
        gradients = np.gradient(initial_set)
        if ((sum(gradients) / len(gradients)))  > -1 and (initial_set[window_size-1] < 20):
          initial_set = initial
        
    return result




import tensorflow as tf

def LSTM_step(cell_inputs, cell_states, kernel, recurrent_kernel, bias):
    """
    Run one time step of the cell. That is, given the current inputs and the cell states from the last time step, calculate the current state and cell output.
    You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.
    Hint: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.
        
        
    :param cell_inputs: The input at the current time step. The last dimension of it should be 1.
    :param cell_states:  The state value of the cell from the last time step, containing previous hidden state h_tml and cell state c_tml.
    :param kernel: The kernel matrix for the multiplication with cell_inputs
    :param recurrent_kernel: The kernel matrix for the multiplication with hidden state h_tml
    :param bias: Common bias value
    
    
    :return: current hidden state, and a list of hidden state and cell state. For details check TensorFlow LSTMCell class.
    """
    
    
    ###################################################
    # TODO:      INSERT YOUR CODE BELOW               #
    # params                                          #
    ###################################################
    
    print('./utils/LSTM_step.py not implemented!') # delete me
    
    ###################################################
    # END TODO                                        #
    ###################################################

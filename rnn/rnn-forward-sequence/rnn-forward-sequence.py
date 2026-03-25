import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch, seq_len, input_size = X.shape
    h_t = h_0
    h_all = []

    for t in range(seq_len):
        x_t = X[:, t, :]         
        h_t = np.tanh(x_t @ W_xh.T + h_t @ W_hh.T + b_h)  
        h_all.append(h_t)

    h_all   = np.stack(h_all, axis=1)  
    h_final = h_all[:, -1, :]          

    return h_all, h_final
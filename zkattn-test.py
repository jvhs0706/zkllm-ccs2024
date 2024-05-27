import numpy as np
import torch
import fileio_utils
import os, sys

if __name__ == '__main__':
    m, n, d = 2048, 2048, 5120
    output_file = 'output.bin'

    Q = torch.randn(m, d, device = 0, dtype = torch.float16).float() * 0.01
    fileio_utils.save_int(Q, 1 << 16, 'Q.bin') 
    K = torch.randn(n, d, device = 0, dtype = torch.float16).float() * 0.01
    fileio_utils.save_int(K, 1 << 16, 'K.bin') 
    V = torch.randn(n, d, device = 0, dtype = torch.float16).float()
    fileio_utils.save_int(V, 1 << 16, 'V.bin') 

    compilation_error = os.system('make test')
    if compilation_error:
        print("Error compiling test")
        exit(1)

    os.system(f'./test Q.bin K.bin V.bin {m} {n} {d} output.bin')
    Y = torch.tensor(np.fromfile('output.bin', dtype = np.int32).reshape(n, d), device = 0) / (1 << 16)
    rowwise_sum = Y.sum(axis = 1)

    Y_comp = torch.nn.functional.softmax((Q @ K.T) / np.sqrt(d), dim = 1) @ V
    Y_comp_ = torch.nn.functional.softmax((Q.half() @ K.half().T) / np.sqrt(d), dim = 1) @ V.half()
    Y_half = Y_comp.half()
    print(f'Converted to half: {torch.abs(Y_comp - Y_half).mean().item()}, {torch.abs(Y_comp - Y_half).max().item()}')
    print(f'The maximum error is: {torch.abs(Y_comp - Y).mean().item()}, {torch.abs(Y_comp - Y).max().item()}')
    print(f'The real half error is : {torch.abs(Y_comp - Y_comp_).mean().item()}, {torch.abs(Y_comp - Y_comp_).max().item()}')


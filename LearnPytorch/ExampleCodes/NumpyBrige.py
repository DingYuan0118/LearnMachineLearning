import torch
import numpy as np

a = torch.ones(5,3)
b = a.numpy()

a_f = np.ones(5, 3)
b_f = torch.from_numpy(a_f)
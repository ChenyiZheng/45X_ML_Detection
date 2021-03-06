import torch
import time


def write_logs(filename, line):
    with open(filename + '.txt', 'a') as f:
        f.write(line + '\n')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
import numpy as np


def stop_training_func(data_list, min_iter_to_check=100, epoch_dist=200, gen_label="generator_loss"):
    if len(data_list) < min_iter_to_check:
        return False
    
    epoch_array = np.array([d["epoch"] for d in data_list])
    generator_array = np.array([d[gen_label] for d in data_list])
    
    min_epoch_loss = epoch_array[np.argmin(generator_array)]
    return True if (epoch_array[-1] - min_epoch_loss) > epoch_dist else False

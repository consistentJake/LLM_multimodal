
from datetime import datetime
import numpy as np

def calculate_weights(_labels, is_use_sqrt=False):
    labels_np = _labels.detach().numpy()
    num_datapoint = _labels.shape[0]
    pos_num = (np.array(labels_np[:num_datapoint])==True).sum()
    neg_num = (np.array(labels_np[:num_datapoint])==False).sum()
    if is_use_sqrt:
        calc_weight = lambda x, y: np.sqrt(1 - (x/y))
    else:
        calc_weight = lambda x, y: 1 - (x/y)
    weight_dict = {0:calc_weight(neg_num, num_datapoint),1:calc_weight(pos_num, num_datapoint)}
    return pos_num, neg_num, num_datapoint, weight_dict

def get_fp_rate(conf_matrix):
    tn = conf_matrix[0, 0].item()  # True Negative
    fp = conf_matrix[0, 1].item()  # False Positive
    # fn = conf_matrix[1, 0].item()  # False Negative (not used in this calculation)
    # tp = conf_matrix[1, 1].item()  # True Positive (not used in this calculation)
    
    # Calculate the false positive rate
    fp_rate = fp / (fp + tn)
    return fp_rate

def get_accuracy(conf_matrix):
    tn = conf_matrix[0, 0].item()  # True Negative
    fp = conf_matrix[0, 1].item()  # False Positive
    fn = conf_matrix[1, 0].item()  # False Negative
    tp = conf_matrix[1, 1].item()  # True Positive
    
    # Calculate the accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy

def get_formatted_time():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as 'YYYY_MM_DD_HH_mm'
    formatted_time = now.strftime('%Y_%m_%d_%H_%M')

    return formatted_time

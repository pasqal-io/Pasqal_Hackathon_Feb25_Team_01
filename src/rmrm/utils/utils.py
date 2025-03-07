import os
import datetime as dt
import json
import collections
import re
import torch 
from lifelines.utils import concordance_index


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
    

def make_dir(dir_path):
    """
    Make directory if doesn't exist
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def delete_file(path):
    """
    Delete file if exists
    """
    if os.path.exists(path):
        os.remove(path)


def get_files_list(path, ext_array=['.tif']):
    """
    Get all files in a directory with a specific extension
    """
    files_list = list()
    dirs_list = list()

    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if any(x in file for x in ext_array):
                files_list.append(os.path.join(root, file))
                folder = os.path.dirname(os.path.join(root, file))
                if folder not in dirs_list:
                    dirs_list.append(folder)

    return files_list, dirs_list
    

def json_file_to_pyobj(filename):
    """
    Read json config file
    """
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def get_experiment_id(make_new, load_dir, fold_id):
    """
    Get timestamp ID of current experiment
    """    
    if make_new is False:
        if load_dir == 'last':
            folders = next(os.walk('experiments'))[1]
            folders = [x for x in folders if ('fold' + str(fold_id) + '_') in x]
            folders = sorted_alphanumeric(folders)
            folder_last = folders[-1]
            timestamp = folder_last.replace('\\','/')
        else:
            timestamp = load_dir
    else:
        timestamp = 'fold' + str(fold_id) + '_' + dt.datetime.now().strftime("%Y_%B_%d_%H_%M_%S")
    
    return timestamp


def PartialLogLikelihood(logits, death_indicator, ties='noties'):
    """
    Compute loss for surival prediction

    death_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    """

    logL = 0
    # pre-calculate cumsum
    # cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * death_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(death_indicator, 0)
    final_loss = 1.0*logL / observations

    return final_loss
    

def calc_concordance_index(logits, death_indicator, death_time):
    """
    Compute C-index
    """
    hr_pred = -logits 
    ci = concordance_index(death_time,
                            hr_pred,
                            death_indicator)
    return ci
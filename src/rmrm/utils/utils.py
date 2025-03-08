import os
import datetime as dt
import json
import collections
import re 
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


def get_experiment_id(make_new, load_dir):
    """
    Get timestamp ID of current experiment
    """    
    if not make_new:
        if load_dir == 'last':
            folders = next(os.walk('results'))[1]
            folders = sorted_alphanumeric(folders)                   
            timestamp = folders[-1] if folders else None
        else:
            timestamp = load_dir
    else:
        timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    return timestamp
    

def calc_concordance_index(logits, death_indicator, death_time):
    """
    Compute C-index
    """
    hr_pred = -logits 
    ci = concordance_index(death_time,
                            hr_pred,
                            death_indicator)
    return ci
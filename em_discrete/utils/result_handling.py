import pandas as pd
import yaml

import os


def parse_directory(path,
                    hparam_names=None):
    """
    Parse the directory at the given path with all the learned models
    following columns:
        - 'path': path to the saved models (dictionary with all the version_<run_id> files)
        - <hparam_name>: <hparam_value>

    Input:
        - path: path to the directory with the saved models
        - hparam_names: list of hyperparameter names to include in the dataframe

    Output:
        - df: pandas dataframe with all the hyperparameters and the path to the saved models
    """

    if hparam_names is None:
        hparam_names = ['batch_size', 'curriculum', 'curriculum_threshold', 'hidden_dim', 'input_dim',
                        'l2_penalty', 'learning_rate', 'seed', 'seq_length', 'task_id']

    # Get all the files in the directory
    result_dict = {key: [] for key in hparam_names}
    result_dict['path'] = []

    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

        def tuple(self, node):
            ## construct a tupel from yaml node
            # print(self.construct_sequence(node))
            return tuple(self.construct_sequence(node))
            # return None

    SafeLoaderIgnoreUnknown.add_constructor(u'tag:yaml.org,2002:python/tuple', SafeLoaderIgnoreUnknown.tuple)
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    for filename in os.listdir(path):
        hparam_file = os.path.join(path, filename, 'hparams.yaml')
        if os.path.isfile(hparam_file):
            with open(hparam_file, 'r') as f:
                # add model path to the dictionary
                try:
                    checkpoint_files = os.listdir(os.path.join(path, filename, 'checkpoints'))
                except FileNotFoundError:
                    continue
                if len(checkpoint_files) > 0:
                    checkpoint_files.sort()
                    result_dict['path'].append(os.path.join(path, filename, 'checkpoints', checkpoint_files[-1]))

                    # add all hyperparameters in hparam_names to the dictionary
                    hparams = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)
                    for key in hparam_names:
                        result_dict[key].append(hparams[key])

    return pd.DataFrame.from_dict(result_dict)

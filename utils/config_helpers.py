import mlflow

def flatten_config(config, pre=''):
    if not type(config) is dict:
        raise ValueError("A dictionary should be prvided for this method to work properly!")
    else:
        out_dict = {}
        for key, val in config.items():
            if type(val) is dict:
                out_dict.update(flatten_config(val, pre=f"{pre}.{key}"))
            elif type(val) is list:
                out_dict[f"{pre}.{key}"] = '_'.join([str(i) for i in val])
            else:
                # It is assumed that the result will now be a str, int or float
                out_dict[f"{pre}.{key}"] = val
        return out_dict

def log_config(config):
    flat_config = flatten_config(config)
    for key, val in flat_config.items():
        mlflow.log_param(key, val)
            
if __name__ == '__main__':
    print(flatten_config(test))
    print(flatten_config({}))
    print(flatten_config({'a': {'b': {'c': 1, 'd': '2'}}}))
    print(flatten_config([1, 2, 3]))
    print(flatten_config)
    

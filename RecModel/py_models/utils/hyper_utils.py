def unfold_config(cfg):
    new = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            tmp = unfold_config(val)
            res = tmp.pop('type')
            tmp[key] = res
            new.update(tmp)
        else:
            new.update({key: val})
    return new
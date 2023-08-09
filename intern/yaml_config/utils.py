import collections
import yaml


def yaml_to_env(config_file: str) -> str:
    """_summary_

    Args:
        config_file (str): _description_

    Returns:
        str: _description_
    """
    config_file = yaml.safe_load(config_file)
    env_file = ""

    def rec(data, parent_key=None):
        nonlocal env_file
        for k, v in data.items():
            k = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                rec(data=v, parent_key=k)
            else:
                env_file += f"{k}={v}\n"

    rec(data=config_file)
    return env_file


def env_to_yaml(env_list: str) -> str:
    """_summary_

    Args:
        config_file (str): _description_

    Returns:
        str: _description_
    """
    yaml_file = {}
    env_list = [i for i in env_list.split("\n") if i]

    def is_digit(value):
        try:
            value = float(value)
            return value
        except ValueError:
            return False

    def rec(data):
        keys = data.split("=")[0]
        v = data.split("=")[1]
        if len(keys.split(".")) == 1:
            if v.isdigit():
                v = int(v)
            elif is_digit(v):
                v = is_digit(v)
            elif v in ["true", "True"]:
                v = True
            elif v in ["false", "False"]:
                v = False
            return {keys: v}
        k = keys.split(".")[0]
        data = f'{".".join(keys.split(".")[1:])}={v}'
        return {k: (rec(data))}

    def update(d, upd_d):
        for k, v in upd_d.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    for idx, el in enumerate(env_list):
        env_list[idx] = rec(el)
    for el in env_list:
        yaml_file = update(yaml_file, el)
    yaml_file = yaml.safe_dump(yaml_file)
    return yaml_file

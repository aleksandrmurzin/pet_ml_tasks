import yaml
from collections import defaultdict

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
        env_list (str): _description_

    Returns:
        str: _description_
    """
    yaml_list = []
    yaml_file = {}
    env_list = [i for i in env_list.split("\n") if i]


    def rec(keys, value):
        nonlocal yaml_file
        if len(keys.split(".")) == 1:
            if value.isdigit():
                value = int(value)
            elif value.isdecimal():
                value = float(value)
            elif value in ["True", "true"]:
                value = True
            elif value in ["False", "false"]:
                value = False
            return {keys:value}
        return {keys.split(".")[0]: rec(".".join(keys.split(".")[1:]), value)}
    for elem in sorted(env_list):
        keys = elem.split("=")[0]
        value = elem.split("=")[1]
        kv = rec(keys, value)
        yaml_file.update(kv)

    import pdb; pdb.set_trace()
    yaml_file = yaml.dump(yaml_file)
    return yaml_file

config = """

preprocess_params:
  sr: 24000
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

model_params:
  dim_in: 64
  style_dim: 64
  latent_dim: 16
  num_domains: 20
  max_conv_dim: 512
  n_repeat: 4
  w_hpf: 0
  F0_channel: 256

loss_params:
  g_loss:
    lambda_sty: 1.
    lambda_cyc: 5.
    lambda_ds: 1.
    lambda_norm: 1.
    lambda_asr: 10.
    lambda_f0: 5.
    lambda_f0_sty: 0.1
    lambda_adv: 2.
    lambda_adv_cls: 0.5
    norm_bias: 0.5
  d_loss:
    lambda_reg: 1.
    lambda_adv_cls: 0.1
    lambda_con_reg: 10.
  
  adv_cls_epoch: 50
  con_reg_epoch: 30

optimizer_params:
  lr: 0.0001
"""

config_env_to_yaml = utils
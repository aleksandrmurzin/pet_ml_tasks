import utils
import yaml.loader

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

config_env_to_yaml = utils.yaml_to_env(config)

def test_yaml_to_env():
  
    """_summary_

    Returns
    -------
    _type_
        _description_
    """

    result = utils.yaml_to_env(config_file=config)



def test_env_to_yaml():
    result = utils.env_to_yaml(env_list=config_env_to_yaml)

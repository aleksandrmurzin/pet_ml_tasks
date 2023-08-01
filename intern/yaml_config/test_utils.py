import utils
import yaml.loader

config = """
log_dir: "Models/VCTK20"
save_freq: 2
device: "cuda"
epochs: 150
batch_size: 5
pretrained_model: ""
load_only_params: false
fp16_run: true

train_data: "Data/train_list.txt"
val_data: "Data/val_list.txt"

F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00100.pth"

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

def test_yaml_to_env():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
    # with open("config.yaml", "r") as fi:
    #     config = yaml.safe_load(fi)

    utils.yaml_to_env(config_file=config)
    return None
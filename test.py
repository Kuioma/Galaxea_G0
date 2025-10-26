import torch
import yaml
import helix
import helix.vlm
import helix.vlm.backbone

file_path = "/Users/shijunjie/code/my_porject/Galaxea_G0/helix/cfg/vlm.yml"
with open(file_path, 'r', encoding='utf-8') as file:
    cfg_dict = yaml.safe_load(file)

vlm = helix.vlm.backbone(cfg_dict)


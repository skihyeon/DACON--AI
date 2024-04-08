import torch
import numpy as np
import random
import os
import wandb


def wandb_login():
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    return wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def update_wandb_config(wandb, config_class, prefix=''):
    for attribute_name in dir(config_class):
        if attribute_name.startswith('__'):
            continue
        attribute_value = getattr(config_class, attribute_name)

        # 서브클래스를 돌며 재귀적으로 처리
        if isinstance(attribute_value, type):
            update_wandb_config(wandb, attribute_value, prefix=f"{prefix}{attribute_name}_")
        else:
            config_key = f"{prefix}{attribute_name}"
            setattr(wandb.config, config_key, attribute_value)

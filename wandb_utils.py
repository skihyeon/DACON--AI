import wandb
import os

def wandb_init(project_name, entity_name):
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb.login(key=wandb_api_key)
    wandb.init(project=project_name, entity=entity_name)
    return wandb
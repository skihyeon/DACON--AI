import torch

name = "gpu"

class Config:
    class general:
        seed = 456
        device = "cuda" if torch.cuda.is_available() else "cpu"

        project_name_pt = f"deepSVDD_pretrain_{name}"
        project_name = f"deepSVDD_{name}"
        entity_name = "hero981001"
        
    class train:
        train_csv = './datas/train.csv'
        model_save_path = f'./model_files/deepSVDD/v1_{name}.pt'
        load_batch_size = 16

    class test:
        test_csv = './datas/test.csv'
    
    class model:
        class deepSVDD:
            pretrained = True
            pretrained_path = f'./model_files/deepSVDD/pretrained_params_{name}.pth'
            num_epochs=50
            lr=1e-4
            weight_decay=0.5e-6
            pretrain=True
            step_size = 20
            gamma = 0.1
        
        class C_AE:
            latent_dim = 2048
            num_epochs = 100
            weight_decay = 0.5e-3
            lr = 1e-3
            lr_milestones=[50]
            step_size = 10
            gamma = 0.99
            save_params_path = f'./model_files/deepSVDD/pretrained_params_{name}.pth'
            save_model_path = f'./model_files/deepSVDD/C_AE_{name}.pt'

    
    class inference:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_path = f'./model_files/deepSVDD/pretrained_params_{name}.pth'
        model_path = f'./model_files/deepSVDD/v1_{name}.pt'
        test_file = "./datas/test.csv"
        batch_size = 64
        save_results_path = f"./results/v1_{name}_result.csv"
        submit_path = f'./submits/v1_{name}_submit.csv'
import torch

class Config:
    class general:
        seed = 456
        device = "cuda" if torch.cuda.is_available() else "cpu"

        project_name_pt = "deepSVDD_pt"
        project_name = "deepSVDD"
        entity_name = "hero981001"
        
    class train:
        train_csv = './datas/train.csv'
        model_save_path = './model_files/deepSVDD/v1.pt'
        load_batch_size = 16

    class test:
        test_csv = './datas/test.csv'
    
    class model:
        class deepSVDD:
            pretrained = True
            pretrained_path = './model_files/deepSVDD/pretrained_params.pth'
            num_epochs=5
            lr=1e-4
            weight_decay=0.5e-6
            pretrain=True
            step_size = 10
            gamma = 0.1
        
        class C_AE:
            latent_dim = 512
            num_epochs = 5
            weight_decay = 0.5e-3
            lr = 1e-3
            lr_milestones=[50]
            step_size = 10
            gamma = 0.1
            save_path = './model_files/deepSVDD/pretrained_params.pth'

    
    
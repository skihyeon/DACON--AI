import torch

class Config:
    class general:
        seed = 456
        device = "cuda" if torch.cuda.is_available() else "cpu"

        project_name_pt = "deepSVDD_pretrain_new"
        project_name = "deepSVDD_new"
        entity_name = "hero981001"
        
    class train:
        train_csv = './datas/train.csv'
        model_save_path = './model_files/deepSVDD/v1_test.pt'
        load_batch_size = 16

    class test:
        test_csv = './datas/test.csv'
    
    class model:
        class deepSVDD:
            pretrained = True
            pretrained_path = './model_files/deepSVDD/pretrained_params_test.pth'
            num_epochs=50
            lr=1e-4
            weight_decay=0.5e-6
            pretrain=True
            step_size = 20
            gamma = 0.1
        
        class C_AE:
            latent_dim = 512
            num_epochs = 100
            weight_decay = 0.5e-3
            lr = 1e-3
            lr_milestones=[50]
            step_size = 10
            gamma = 0.99
            save_params_path = './model_files/deepSVDD/pretrained_params_test.pth'
            save_model_path = './model_files/deepSVDD/C_AE_test.pt'

    
    class inference:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_path = './model_files/deepSVDD/pretrained_params_test.pth'
        model_path = './model_files/deepSVDD/v1_test.pt'
        test_file = "./datas/test.csv"
        batch_size = 64
        save_results_path = "./results/v1_test_result.csv"
        submit_path = './submits/v1_test_submit.csv'
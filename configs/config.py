class Config:

    class wandb:
        entity = 'hero981001'
        project_name = 'deepSVDD_split'
        run_name = 'v3_deconv_stride'
    class pretrain_molding:
        run_name = 'pretrain_molding'
        z_dim = 2240
        lr = 0.1
        weight_decay = 0
        step_size = 50
        gamma = 0.1
        epochs = 50
        save_path = './model_files/deepSVDD_split/'
        save_model_path = save_path + f'{run_name}.pt'
    
    class pretrain_leadframe:
        run_name = 'pretrain_leadframe'
        z_dim = 2240
        lr = 0.001
        weight_decay = 0
        step_size = 20
        gamma = 0.5
        epochs = 50
        save_path = './model_files/deepSVDD_split/'
        save_model_path = save_path + f'{run_name}.pt'
    
    class train_molding:
        run_name = 'train_molding'
        z_dim = 2240
        param_path = './model_files/deepSVDD_split/params_molding.pth'
        pretrained = True
        lr = 0.001
        weight_decay = 0
        step_size = 20
        gamma = 0.5
        epochs = 100
        save_path = './model_files/deepSVDD_split/'
        save_model_path = save_path + f'{run_name}.pt'
        img_path = './datas/train_molding.csv'
        batch_size = 16
    
    class train_leadframe:
        run_name = 'train_leadframe'
        z_dim = 2240
        param_path = './model_files/deepSVDD_split/params_leadframe.pt'
        pretrained = True
        lr = 0.001
        weight_decay = 0
        step_size = 20
        gamma = 0.5
        epochs = 100
        save_path = './model_files/deepSVDD_split/'
        save_model_path = save_path + f'{run_name}.pt'
        img_path = './datas/train_leadframe.csv'
        batch_size = 16
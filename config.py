class Config:
    seed = 456
    batch_size = 64
    num_epochs = 1000
    learning_rate = 0.01
    step_size = 10
    gamma = 0.5
    device = "cuda"
    project_name = "Segment_Anomaly_Detection"
    entity_name = "hero981001"
    train_csv = './datas/train.csv'
    test_csv = './datas/test.csv'
    model_save_path = './model_files/AE_v3.pt'
    
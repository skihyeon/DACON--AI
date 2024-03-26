class Config:
    seed = 456
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = "cuda"
    project_name = "Segment_Anomaly_Detection"
    entity_name = "hero981001"
    train_csv = './datas/train.csv'
    model_save_path = './model_files/AE.pt'
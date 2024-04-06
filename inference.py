import torch
import numpy as np
from data_loader import get_test_loader
from config import Config
from tqdm.auto import tqdm
import pandas as pd

def load_c(pretrained_path):
    AE_state_dict = torch.load(pretrained_path)
    return torch.Tensor(AE_state_dict['center'])

def perform_anomaly_detection(test_loader, model_path, c, device):
    model = torch.jit.load(model_path).to(device)

    model.eval()

    anomaly_scores = []
    with torch.no_grad():
        for x, _ in tqdm(test_loader):
            x = x.float().to(device)
            z = model(x)
            score = torch.sum((z - c) ** 2, dim=1)
            anomaly_scores.append(score)
    anomaly_scores = torch.cat(anomaly_scores).cpu().numpy()
    
    return anomaly_scores

def define_threshold(anomaly_scores, method='quantile', q=0.95):
    if method == 'quantile':
        return np.quantile(anomaly_scores, q)
    elif method == 'mean_std':
        return anomaly_scores.mean() + 3 * anomaly_scores.std()
    else:
        raise ValueError("Unknown method for threshold definition")
    

def save_results(anomaly_scores, predictions, result_path, submit_path):
    with open(result_path, 'w') as f:
        for score, pred in zip(anomaly_scores, predictions):
            f.write(f'{score},{pred}\n')

    sub = pd.read_csv('./datas/sample_submission.csv')
    sub['label'] = predictions
    sub.to_csv(submit_path, index=False)
    print(f'Results saved to {result_path}')


def main():
    config = Config()
    device = config.inference.device
    model_path = config.inference.model_path 
    csv_file = config.inference.test_file
    batch_size = config.inference.batch_size

    test_loader = get_test_loader(csv_file, batch_size)

    center = load_c(config.inference.pretrained_path)

    anomaly_scores = perform_anomaly_detection(test_loader, model_path, center, device)

    threshold = define_threshold(anomaly_scores)

    predictions = (anomaly_scores > threshold).astype(int)

    save_results(anomaly_scores, predictions, config.inference.save_results_path, config.inference.submit_path)

if __name__ == '__main__':
    main()
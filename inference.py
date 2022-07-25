import os, os.path

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

from utils import *

import configparser
import torch
import pandas as pd

config = configparser.ConfigParser()
config.read('config.ini')

test_path = config.get('DEFAULT', 'test_path')
n_folds = config.getint('DEFAULT', 'n_folds')
n_labels = config.getint('DEFAULT', 'n_labels')
batch_size = config.getint('DEFAULT', 'batch_size')
hidden_neurons = config.getint('DEFAULT', 'hidden_neurons')

test_data = pd.read_csv(test_path, encoding='utf-8')
test_data['Текст Сообщения'] = test_data['Текст Сообщения'].apply(cleanhtml)
test_data['Текст Сообщения'] = test_data['Текст Сообщения'].apply(lambda x: x[:512])
test_data['Текст Сообщения'] = test_data['Текст Сообщения'].apply(lambda x: '<s> ' + x + ' </s>')
x_test = test_data['Текст Сообщения']
    
test_dataloader = create_dataloader(x_test, batch_size, 'sequential')

models_dir = './roberta_models/'

all_test_probs = torch.empty([n_folds, x_test.shape[0], n_labels])

for i, f_model_name in enumerate(os.listdir(models_dir)):
    model_path = os.path.join(models_dir, f_model_name)
    model = myBertModel(hidden_neurons=hidden_neurons, n_labels=n_labels)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    test_probs = do_test_preds(model, test_dataloader)
    all_test_probs[i, :, :] = torch.tensor(test_probs)

test_preds = all_test_probs.mean(axis=0).argmax(axis=1)
test_data['Категория'] = test_preds

test_data.to_csv('./data/prediction_test_data.csv', index=False)
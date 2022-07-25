import os, os.path

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

from utils import *

import torch
import numpy as np
import pandas as pd
import re
import configparser

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import Adam, AdamW
from torch.nn import functional as F

def main():
    global_seed = 144
    seed_step = 100
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    train_path = config['DEFAULT']['train_path']
    test_path = config['DEFAULT']['test_path']

    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')

    train_data = train_data[train_data['Категория'] != 12]

    train_data['Текст Сообщения'] = train_data['Текст Сообщения'].apply(cleanhtml)
    test_data['Текст Сообщения'] = test_data['Текст Сообщения'].apply(cleanhtml)

    train_data['Текст Сообщения'] = train_data['Текст Сообщения'].apply(lambda x: x[:512])
    test_data['Текст Сообщения'] = test_data['Текст Сообщения'].apply(lambda x: x[:512])

    train_data['Текст Сообщения'] = train_data['Текст Сообщения'].apply(lambda x: '<s> ' + x + ' </s>')
    test_data['Текст Сообщения'] = test_data['Текст Сообщения'].apply(lambda x: '<s> ' + x + ' </s>')

    repl_dict = {key:i for i, (key, value) in enumerate(dict(train_data['Категория'].value_counts()).items())}
    train_data['Категория'] = train_data['Категория'].map(repl_dict)

    x_train, y_train = train_data['Текст Сообщения'], train_data['Категория']
    x_test = test_data['Текст Сообщения']

    n_labels = y_train.nunique()
    config.set('DEFAULT', 'n_labels', str(n_labels))

    with open("config.ini", "w") as f:
        config.write(f)

    batch_size = config.getint('DEFAULT', 'batch_size')
    n_folds = config.getint('DEFAULT', 'n_folds')
    n_epochs = config.getint('DEFAULT', 'n_epochs')
    val_roc_auc_treshold = config.getfloat('DEFAULT', 'val_roc_auc_treshold')

    folds = StratifiedKFold(n_splits=n_epochs, shuffle=True, random_state=global_seed)
    splits = folds.split(x_train, y_train)
    y_probs = np.zeros([x_test.shape[0], n_labels])
    y_oof = np.zeros(x_train.shape[0])
    f1 = 0
    roc_auc = 0

    test_dataloader = create_dataloader(x_test, batch_size, 'sequential')

    for fold_n, (train_index, valid_index) in enumerate(splits):
        global_seed += fold_n * seed_step
        set_seed(global_seed)
        X_tr, X_val = x_train.iloc[train_index].to_numpy(), x_train.iloc[valid_index].to_numpy()
        y_tr, y_val = y_train.iloc[train_index].to_numpy(), y_train.iloc[valid_index].to_numpy()

        train_dataloader = create_dataloader(X_tr, batch_size, 'sequential', y_tr)
        val_dataloader = create_dataloader(X_val, batch_size, 'sequential', y_val)
        print(f"Fold [{fold_n + 1}/{n_folds}]")

        model, val_preds, val_roc_auc = train_val_loop(train_dataloader, val_dataloader, n_epochs, fold_n)
        y_test_probs = do_test_preds(model, test_dataloader)

        candidates_val_preds = [val_preds]
        candidates_test_probs = [y_test_probs]
        candidates_val_roc_aucs = np.array(val_roc_auc)

        n_rep = 0
        while val_roc_auc <= val_roc_auc_treshold:
            if n_rep >= 5:
                break
            print(f"Fold [{fold_n + 1}/{n_folds}] is repeated")
            model, val_preds, val_roc_auc = train_val_loop(train_dataloader, val_dataloader, n_epochs, fold_n)
            y_test_probs = do_test_preds(model, test_dataloader)

            candidates_val_preds.append(val_preds)
            candidates_test_probs.append(y_test_probs)
            candidates_val_roc_aucs = np.append(candidates_val_roc_aucs, val_roc_auc)
            n_rep += 1

        max_ind_val_roc_auc = candidates_val_roc_aucs.argmax()
        val_preds = candidates_val_preds[max_ind_val_roc_auc]
        y_test_probs = candidates_test_probs[max_ind_val_roc_auc]

        y_oof[valid_index] = val_preds
        f1 += f1_score(y_val, val_preds, average='weighted') / n_folds

        map_labels = {real_label: i for i, real_label in enumerate(set(y_val))}
        converted_y_val = [map_labels[val] for val in y_val]
        converted_val_preds = [map_labels[val] for val in val_preds]

        roc_auc += roc_auc_score(converted_y_val, one_label_to_many(converted_val_preds, len(map_labels)), multi_class='ovo') / n_folds
        y_probs += y_test_probs / n_folds
        del X_tr, X_val, y_tr, y_val
        model.to('cpu')

    print(f"\nMean F1 = {f1}")
    print(f"\nOut of folds F1 = {f1_score(y_train, y_oof, average='weighted')}")
    print(f"\nMean ROC-AUC = {roc_auc}")
    print(f"\nOut of folds ROC-AUC = {roc_auc_score(y_train, one_label_to_many(y_oof, n_labels),  multi_class='ovo')}")

    test_preds = y_probs.argmax(axis=1)
    reverse_repl_dict = {value:key for key, value in repl_dict.items()}
    test_data['Категория'] = test_preds
    test_data['Категория'] = test_data['Категория'].map(reverse_repl_dict)
    n_submission = max([int(file.split('_')[-1].split('.')[0]) for file in os.listdir('./submissions') if 'submission' in file]) + 1
    test_data[['id', 'Категория']].to_csv(f'./data/submission_{n_submission}.csv', index=False)
    
if __name__ == '__main__':
    main()
import torch
import numpy as np
import pandas as pd
import re
import os, os.path
import configparser
import random

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, roc_auc_score
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.optim import Adam, AdamW
from torch.nn import functional as F

config = configparser.ConfigParser()
config.read('config.ini')

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

tokenizer = AutoTokenizer.from_pretrained(config['DEFAULT']['model_name'])
loss = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class myBertModel(torch.nn.Module):
    def __init__(self, hidden_neurons, n_labels):
        super(myBertModel, self).__init__()
        self.n_labels = n_labels
        bert_config = AutoConfig.from_pretrained(config['DEFAULT']['model_name'], num_labels=self.n_labels)
        self.model = AutoModel.from_pretrained(config['DEFAULT']['model_name'], config=bert_config)
        self.hidden_neurons = hidden_neurons
        self.linear1 = torch.nn.Linear(bert_config.hidden_size, int(self.hidden_neurons / 2))
        self.classifier = torch.nn.Linear(bert_config.hidden_size, self.n_labels)
        self.dropout = torch.nn.Dropout(.3)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(bert_config.hidden_size, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = bert_output[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        scores = self.classifier(feature)
        return scores

def set_seed(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, ' ', raw_html)
    cleantext = re.sub('\s+', ' ', cleantext)
    return cleantext

def create_dataloader(x_data, batch_size, sampler, y_data = ''):
    tokenized_texts = [tokenizer.tokenize(sent) for sent in x_data]
    inputs = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    padded_inputs = pad_sequences(inputs, 
                          dtype='long', 
                          padding='post', 
                          truncating='post')
    masks = [[float(i>0) for i in seq] for seq in padded_inputs]
    
    padded_inputs = torch.tensor(padded_inputs)
    masks = torch.tensor(masks)
    
    try:
        labels = np.array(y_data)
        labels = torch.tensor(labels)
        tensor_dataset = torch.utils.data.TensorDataset(padded_inputs, masks, labels)
    except:
        tensor_dataset = torch.utils.data.TensorDataset(padded_inputs, masks)
    
    if sampler == 'random':
        dataloader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(tensor_dataset)
        )
    elif sampler == 'sequential':
        dataloader = torch.utils.data.DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SequentialSampler(tensor_dataset)
        )
    return dataloader

def train_loop(train_dataloader, val_dataloader, model, n_epochs, fold):
    optimizer = AdamW(model.parameters(), lr=config.getfloat('DEFAULT', 'lr'))
    num_train_steps = int(len(list(train_dataloader)) * n_epochs)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=60, num_training_steps=num_train_steps
    )
    
    prev_val_roc_auc = 0
    lrs = []
    
    for epoch in range(n_epochs):
        
        model.train()
        epoch_loss_set = []
        
        optimizer.zero_grad()

        for j, batch in enumerate(tqdm(train_dataloader)):
            
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            pred_probs = model.forward(b_input_ids, b_input_mask)
            
            my_loss = loss(pred_probs, b_labels)
            epoch_loss_set.append(my_loss.item())
            my_loss.backward()
                 
            pred_probs = pred_probs.detach().to('cpu').numpy()
            pred_labels = pred_probs.argmax(axis=1)
            b_labels = b_labels.to('cpu').numpy()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.getint('DEFAULT', 'max_grad_norm'))
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()
                 
            if j % 20 == 0 and j != 0:
                print('Epoch [{}/{}] ... Step [{}] ... LR: {:.8f} ... Grad: {:.2f} ... Mean loss: {:.4f}'.format(epoch + 1, n_epochs, j, scheduler.get_last_lr()[0], grad_norm, np.mean(epoch_loss_set)))
                print()
                
            if j == 0:
                all_train_labels = b_labels
                all_train_preds = pred_labels
            else:
                all_train_labels = np.hstack([all_train_labels, b_labels])
                all_train_preds = np.hstack([all_train_preds, pred_labels])
                                
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        
        map_labels = {real_label: i for i, real_label in enumerate(set(all_train_labels))}
        converted_all_train_labels = [map_labels[val] for val in all_train_labels]
        converted_all_train_preds = [map_labels[val] for val in all_train_preds]
            
        train_roc_auc = roc_auc_score(converted_all_train_labels, one_label_to_many(converted_all_train_preds, len(map_labels)), multi_class='ovo')
        
        print()
        print('Epoch [{}/{}] ... Train ROC-AUC: {:.2f} ... Train f1: {:.2f} ... Mean train loss: {:.4f}'.format(epoch + 1, n_epochs, train_roc_auc, train_f1, np.mean(epoch_loss_set)))
        print()
        
        all_val_preds, current_val_roc_auc = validation_loop(val_dataloader, model)
        
        if current_val_roc_auc > prev_val_roc_auc:
            prev_val_roc_auc = current_val_roc_auc
            torch.save(model.state_dict(), f'./folds_models/xlm-roberta_model_{fold}_{epoch}.pt')
            continue
        else:
            model.load_state_dict(torch.load(f'./folds_models/xlm-roberta_model_{fold}_{epoch - 1}.pt'))
            break
    return model
        
def validation_loop(val_dataloader, model):
    val_loss_set = []

    model.eval()

    for j, batch in enumerate(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            pred_probs = model.forward(b_input_ids, b_input_mask)
            val_loss = loss(pred_probs, b_labels)
            val_loss_set.append(val_loss.item())

            b_labels = b_labels.to('cpu').numpy()
            pred_probs = pred_probs.detach().to('cpu').numpy()
            pred_labels = pred_probs.argmax(axis=1)

        if j == 0:
            all_val_labels = b_labels
            all_val_preds = pred_labels
        else:
            all_val_labels = np.hstack([all_val_labels, b_labels])
            all_val_preds = np.hstack([all_val_preds, pred_labels])
    
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    
    map_labels = {real_label: i for i, real_label in enumerate(set(all_val_labels))}
    converted_all_val_labels = [map_labels[val] for val in all_val_labels]
    converted_all_val_preds = [map_labels[val] for val in all_val_preds]
    
    val_roc_auc = roc_auc_score(converted_all_val_labels, one_label_to_many(converted_all_val_preds, len(map_labels)), multi_class='ovo')
    print()
    print('Val ROC-AUC {:.2f} ... Val f1: {:.2f} ... Mean val loss: {:.4f}'.format(val_roc_auc, val_f1, np.mean(val_loss_set)))
    print()
    print('-------------------------------------------------------------------------------------------------------------------------')
    print()
    return all_val_preds, val_roc_auc

def do_test_preds(model, test_dataloader):
    model.eval()
    test_probs = torch.empty(0, config.getint('DEFAULT', 'n_labels'))
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            pred_probs = model.forward(b_input_ids, b_input_mask)
            pred_probs = pred_probs.detach().to('cpu')
            test_probs = torch.cat([test_probs, pred_probs])
    return test_probs.numpy()

def train_val_loop(train_dataloader, val_dataloader, n_epochs, fold_n):
    model = myBertModel(hidden_neurons=config.getint('DEFAULT', 'hidden_neurons'), n_labels=config.getint('DEFAULT', 'n_labels'))
    model.to(device)
    model = train_loop(train_dataloader, val_dataloader, model, n_epochs, fold_n)
    val_preds, val_roc_auc = validation_loop(val_dataloader, model)
    return model, val_preds, val_roc_auc

def one_label_to_many(df, n_labels):
    classes = [x for x in range(n_labels)]
    y_test = []
    min_class = min(classes)
    count_class = len(classes)
    for ll in df:
        mass = [0] * count_class
        mass[int(ll) - min_class] = 1
        y_test.append(mass)
    return y_test

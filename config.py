import configparser

config = configparser.ConfigParser()

config['DEFAULT'] = {}

config['DEFAULT']['n_folds'] = '7'
config['DEFAULT']['n_epochs'] = '10'
config['DEFAULT']['hidden_neurons'] = '512'
config['DEFAULT']['batch_size'] = '24'
config['DEFAULT']['max_grad_norm'] = '2'
config['DEFAULT']['num_warmup_steps'] = '60'
config['DEFAULT']['val_roc_auc_treshold'] = '0.74'
config['DEFAULT']['lr'] = '1e-5'
config['DEFAULT']['train_path'] = '../train_dataset_train.csv'
config['DEFAULT']['test_path'] = '../test_dataset_test.csv'
config['DEFAULT']['model_name'] = 'DeepPavlov/xlm-roberta-large-en-ru'

with open('config.ini', 'w') as f:
    config.write(f)
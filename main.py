import numpy as np
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import model
import yaml
import argparse
from model import Unet
import train
from torchvision import transforms
from data.dataset import get_loader

parser = argparse.ArgumentParser(description='Model configuration')
parser.add_argument('--config', default='configs\example.yaml')

_CSV_FILE_ = './data/image_pairs'

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)

    t0 = time.time()

    if config['use_gpu'] and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print('GPU is ready!')
    else:
        device = torch.device('cpu')
        print('CPU only!')
    
    
    # step 1: data   
    # get data loader
    train_data, val_data, test_data = None, None, None
    train_loader, train_data = get_loader(data_dir=config['data_dir'], csv_file=_CSV_FILE_, split='train', batch_size=config['batch_size'],shuffle=True, num_workers=8,max_num_samples=-1)
    val_loader, val_data = get_loader(data_dir=config['data_dir'], csv_file=_CSV_FILE_, split='val', batch_size=config['batch_size'],shuffle=True, num_workers=8,max_num_samples=-1)
    test_loader, test_data = get_loader(data_dir=config['data_dir'], csv_file=_CSV_FILE_, split='test', batch_size=config['batch_size'],shuffle=True, num_workers=8,max_num_samples=-1)

    print("Training samples: ", len(train_data))
    print("Validation samples: ", len(val_data))
    print("Test samples: ", len(test_data))

    # step 2: define the model
    if config['model_name'] == 'Unet':
        model = Unet(config['n_class'])
    else:
        raise ValueError('Invalid model name !')


    train_epoch = getattr(train, 'Train' + config['model_name'])()
    model.to(device)

    print('network structure is shown:\n\n{}'.format(model))

    # step 3: loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduce_factor'],
                                                     patience=config['lr_schedule_patience'], verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []

    # step 4: training
    print('\n start training! \n')
    time.sleep(1)
    with tqdm(range(config['max_epoch'])) as t:
        for epoch in t:
            t.set_description('Epoch:{}'.format(epoch))
            start = time.time()
            epoch_train_loss, optimizer = train_epoch.train(model, optimizer, device, train_loader, epoch)
            epoch_val_loss = train_epoch.validate(model, device, val_loader, epoch)
            epoch_test_loss = train_epoch.validate(model, device, test_loader, epoch)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)

            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'], train_loss=epoch_train_loss,
                          val_loss=epoch_val_loss)
            per_epoch_time.append(time.time() - start)
            scheduler.step(epoch_val_loss)

            # Stop training if max hours exceed
            if time.time() - t0 > config['max_time'] * 3600:
                print('-' * 20)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(config['max_time']))
                break

    # print training/testing information
    time.sleep(1)

    print("Convergence Time (Epochs): {:.4f}".format(epoch + 1))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print('Finish!')


    # Write the results in results folder (TBD)

if __name__ == '__main__':
    main()
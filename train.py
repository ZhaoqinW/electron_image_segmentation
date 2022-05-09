import numpy as np
import os
import torch
import time
from torch.autograd import Variable

"""
functions in class for trianing model in single epoch (simplify the main.py)
"""

class Train(object):
    def train(self, model, optimizer, device, data_loader, epoch):
        model.train()
        epoch_loss = 0
        for it, (img_inputs, target) in enumerate(data_loader):
            t1 = time.time()

            img_inputs = img_inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            predicts = model(img_inputs)
            loss = model.loss(predicts, target,device)
            loss.backward()
            optimizer.step()
            epoch_loss_plus = loss.detach().item()

            epoch_loss = epoch_loss + epoch_loss_plus
            t2 = time.time()
            if (it + 1) % 10 == 0:
                print("Epoch: {} loss: {:.2f}, time: {:.2f}s".format(epoch, epoch_loss_plus, t2 - t1))

        epoch_loss /= (it + 1)

        return epoch_loss, optimizer

    def validate(self, model, device, data_loader, epoch):
        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for it, (img_inputs, target) in enumerate(data_loader):
                img_inputs = img_inputs.to(device)
                target = target.to(device)
                predicts = model(img_inputs)
                loss = model.loss(predicts, target,device)

                epoch_test_loss += loss.detach().item()

            epoch_test_loss /= (it + 1)

        return epoch_test_loss
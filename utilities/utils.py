import torch
import matplotlib.pyplot as plt

def get_mean_and_std(ds):
    imgs = torch.stack([img_t for img_t, _ in ds], dim=3)
    mean, std = imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1)
    return mean, std

def loss_plot(history):
    plt.plot(history['epochs'], history['train_loss'], c= 'b', label = 'training loss')
    plt.plot(history['epochs'], history['test_loss'], c = 'r', label = 'testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = 0)
    plt.show()
    
def acc_plot(history):
    plt.plot(history['epochs'], history['train_acc'], c= 'b', label = 'training acc')
    plt.plot(history['epochs'], history['test_acc'], c = 'r', label = 'testing acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = 0)
    plt.show()
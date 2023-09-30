from tqdm.auto import tqdm
import torch

def fit(epochs, model, train_dl, test_dl, criterion, acc_fn, optimizer, device, step = None):
    torch.manual_seed(42)
    
    train_loss_count = list()
    test_loss_count = list()
    train_acc_count = list()
    test_acc_count = list()
    epoch_count = list()
    
    for epoch in tqdm(range(epochs)):
        print(f'Epoch: {epoch}')

        train_loss, train_acc = .0, .0
        for batch, (X, y) in enumerate(train_dl, 0):
            model.train()
            
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            loss = criterion(y_pred, y)
            train_loss += loss
            
            train_acc += acc_fn(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if batch % 400 == 0:
            #     print(f'looked at {batch * len(X)}/{len(train_dl.dataset)} samples')
                
        train_loss /= len(train_dl)
        train_loss_count.append(train_loss.detach().item())
        train_acc /= len(train_dl)
        train_acc_count.append(train_acc.detach().item() * 100)
        
        
        print(f'Train Loss -> {train_loss:.3f}, Training Accuracy : {train_acc.item() * 100}%')

        test_loss, test_acc = .0, .0
        model.eval()
        with torch.inference_mode():
            for test_X, test_y in test_dl:
                test_X, test_y = test_X.to(device), test_y.to(device)
                
                test_pred = model(test_X)
                
                t_loss = criterion(test_pred, test_y)
                test_loss += t_loss
                
                test_acc += acc_fn(test_pred, test_y)
                
                
            test_loss /= len(test_dl)
            test_loss_count.append(test_loss.detach().item())
            test_acc /= len(test_dl)
            test_acc_count.append(test_acc.detach().item() * 100)
            
        # if epoch % step == 0:
        epoch_count.append(epochs)
        
        print(f'Test Loss -> {test_loss:.3f} | Test accuracy -> {test_acc.item() * 100}%')
        
    return {
        'epochs': epoch_count,
        'train_loss': train_loss_count,
        'test_loss': test_loss_count,
        'train_acc': train_acc_count,
        'test_acc': test_acc_count
    }
                
                
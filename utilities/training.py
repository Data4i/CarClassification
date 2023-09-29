from tqdm.auto import tqdm
import torch

def fit(epochs, model, train_dl, test_dl, criterion, optimizer, device, step = None):
    torch.manual_seed(42)
    
    train_loss_count = list()
    test_loss_count = list()
    epoch_count = list()
    
    for epoch in tqdm(range(epochs)):
        print(f'Epochs: {epoch}')
        epoch_count.append(epochs)
        train_loss = .0
        test_loss = .0
        for batch, (X, y) in enumerate(train_dl, 0):
            model.train()
            
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            loss = criterion(torch.softmax(y_pred, dim = 1).squeeze(), y)
            train_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 400 == 0:
                print(f'looked at {batch * len(X)}/{len(train_dl.dataset)} samples')
                
            train_loss /= len(train_dl)
            train_loss_count.append(train_loss.detach.numpy())
            
        model.eval()
        with torch.inference_mode():
            for X, y in test_dl:
                
                X, y = X.to(device), y.to(device)
                
                test_pred = model(X)
                print(y)
                test_loss = criterion(torch.softmax(test_pred, dim = 1).squeeze(), y.squeeze())
                
                
            test_loss /= len(test_dl)
            test_loss_count.append(test_loss.detach().numpy())
            
        # if epoch % step == 0:
        print(f'Train Loss -> {train_loss:.2f} | Test Loss -> {test_loss:.2f}')
        
    return {
        'train_loss': train_loss_count,
        'test_loss': test_loss_count
    }
                
                
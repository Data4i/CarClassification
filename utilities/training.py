def fit(epochs, model, train_dl, test_dl, criterion, optimizer, device):
    train_loss_count = list()
    test_loss_count = list()
    epoch_count = list()
    
    for epoch in epochs:
        epoch_count.append(epochs)
        model.train()
        train_loss = .0
        test_loss = .0
        for i, batch in enumerate(train_dl, 0):
            inputs, label = batch
            pass
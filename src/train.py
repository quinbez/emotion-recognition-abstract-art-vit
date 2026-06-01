import torch
import torch.nn as nn
import torch.optim as optim

from src.evaluate import evaluate

def get_criterion(class_weights):
    return nn.CrossEntropyLoss(weight=class_weights)

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

def get_scheduler(optimizer, t_max=10):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def train(model, train_loader, val_loader, class_weights, device, epochs=10, save_path=None):
    criterion = get_criterion(class_weights)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history
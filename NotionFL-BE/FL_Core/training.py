# FL_Core/training.py
import torch
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train_model(model, train_loader, epochs, lr, device):
    model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    training_logs = []

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        # Add epoch loss to training logs
        training_logs.append({'epoch': epoch + 1, 'loss': avg_loss})

    return model, training_logs

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_targets.extend(target.view_as(pred).cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    print(f'Test set: Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    return test_loss, accuracy, precision, recall, f1, conf_matrix

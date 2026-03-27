import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SleepTransformer
from preprocess import load_and_clean
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load augmented data
X_train, X_test, y_train, y_test, le_target = load_and_clean('sleep_disorder_dataset_1000 (1).csv')
num_classes = len(le_target.classes_)
print(f"Classes found: {le_target.classes_}")

train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True) # Reduced batch size for synthesized data

test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Initialize the upgraded Transformer
model = SleepTransformer(input_dim=10, num_classes=num_classes).to(device)

# SMOTE already balanced the classes, so we can remove the manual class weights
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(f"Training Transformer on {len(X_train)} augmented samples using device: {device}...")

epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_x, batch_y in progress_bar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item()) 
    
    scheduler.step()
    
    if (epoch+1) % 10 == 0:
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

torch.save(model.state_dict(), 'sleep_model.pth')
print("Complete. Weights saved.")

def evaluate_model(model, loader, device, classes):
    print("\n" + "="*50)
    print("🚀 CALCULATING FINAL METRICS...")
    print("="*50)
    model.eval()  
    all_preds = []
    all_labels = []

    with torch.no_grad():  
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("-" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

evaluate_model(model, test_loader, device, le_target.classes_)
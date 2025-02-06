from main_model import fullmodel
from dataload import train_loader, test_loader
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel
import os
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Initialize tokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = fullmodel(
    embed_size=300,
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
).to(DEVICE)

# Define loss function (ignoring padding token)
criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)

# Define optimizer
LEARNING_RATE = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


def train(epochs, model, train_loader, test_loader, optimizer, criterion, tokenizer):
    """
    Train the image captioning model.
    
    Args:
        epochs (int): Number of training epochs.
        model (torch.nn.Module): Image captioning model.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing captions.
    """
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for idx, (image, caption) in enumerate(train_loader):
            optimizer.zero_grad()
            image, caption = image.to(DEVICE), caption.to(DEVICE)

            # Forward pass
            outputs, attention_weights = model(image, caption)
            targets = caption[:, 1:]  # Shift target for teacher forcing
            outputs_flat = outputs.view(-1, tokenizer.vocab_size)
            targets_flat = targets.view(-1)
            
            # Compute loss
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Compute accuracy ignoring padding
            _, predicted = outputs.max(2)
            mask = targets != tokenizer.pad_token_id
            correct_predictions += (predicted == targets).masked_select(mask).sum().item()
            total_predictions += mask.sum().item()
            
            # Print training stats
            if (idx + 1) % 200 == 0:
                avg_loss = running_loss / 200
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Epoch [{epoch+1}/{epochs}], Step [{idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                running_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
        
        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.inference_mode():
            for image, caption in test_loader:
                image, caption = image.to(DEVICE), caption.to(DEVICE)

                outputs, attention_weights = model(image, caption)
                targets = caption[:, 1:]
                outputs_flat = outputs.view(-1, tokenizer.vocab_size)
                targets_flat = targets.view(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                test_loss += loss.item()
                
                _, predicted = outputs.max(2)
                mask = targets != tokenizer.pad_token_id
                test_correct += (predicted == targets).masked_select(mask).sum().item()
                test_total += mask.sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total if test_total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

# Run training
EPOCHS = 100  # Set desired number of epochs
train(EPOCHS, model, train_loader, test_loader, optimizer, criterion, TOKENIZER)
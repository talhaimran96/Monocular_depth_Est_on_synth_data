from Image_Segmentation.network import R2AttU_Net, AttU_Net
import torch
from torch import nn
# import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import wandb
import albumentations as A
from DepthDataset import Depth_dataset
from tqdm import tqdm
import wandb
import os

resume_training = False

wandb.login(key='5acc1a4862a1e441f2368a1b8c979817fca97071')

dataset_path = dataset_path = "../../../Desktop/Renders/"

model_name = "Attention_Unet"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current Device :{device}")
# define Hyperparameters here

batch_size = 4
epochs = 12
starting_epoch = 0
inital_learing_rate = 0.0001
random_seed = 42
model = AttU_Net().to(device)

# Augmentations
Augmentations = A.Compose(
    [A.RandomBrightnessContrast(p=0.2), A.HorizontalFlip(p=0.5), A.GaussNoise(p=0.2)])

train_dataset = Depth_dataset(dataset_path, train=True, transformations=Augmentations)
test_dataset = Depth_dataset(dataset_path, train=False, transformations=None)

validation_ratio = 0.2
validation_length = int(validation_ratio * len(train_dataset))

# train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
#                                                                   lengths=[len(train_dataset) - validation_length,
#                                                                            validation_length])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# model = smp.Unet(encoder_name=encoder_stage, encoder_weights='imagenet', in_channels=3, classes=1,
#                           activation=None, encoder_depth=5).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=inital_learing_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
loss_function = nn.MSELoss()
final_activation = nn.Sigmoid()

if resume_training:
    checkpoints = torch.load(f'best_depth_model_{model_name}.pth')
    model.load_state_dict(checkpoints['model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['schedular_state_dict'])
    starting_epoch = int(checkpoints['current_epoch'])
    epochs = int(checkpoints['total_epochs'])
    # print((scheduler.get_lr()[0]))

run = wandb.init(
    project="Thesis-1",
    config={
        "Model": model_name,
        "Inital_learning_rate": inital_learing_rate,
        "Epochs": epochs,
        "Batch_size": batch_size,
        "Random Seed": random_seed,
        "optimizer": "Adam",
        "Validation Ratio": validation_ratio,
        "Training Dataset": len(train_dataset)
    }
)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = final_activation(model(data))
        loss = torch.sqrt(criterion(output, target))
        loss.backward()
        optimizer.step()
        wandb.log({"Train_Batch_Loss": loss})

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            output = final_activation(model(data))
            loss = torch.sqrt(criterion(output, target))
            wandb.log({"Validation_Batch_Loss": loss})
            total_loss += loss.item()

    return total_loss / len(val_loader)


# Training loop
best_val_loss = float('inf')
for epoch in range(starting_epoch, epochs):
    train_loss = train(model, train_dataloader, loss_function, optimizer, device)
    val_loss = validate(model, test_dataloader, loss_function, device)
    wandb.log({"Train_epoch_loss": train_loss, "Validation_epoch_loss": val_loss,
               "Current_learning_rate": scheduler.get_lr()[0]})
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    scheduler.step(val_loss)

    # Save the model if the validation loss is the best so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'schedular_state_dict': scheduler.state_dict(), 'current_epoch': epoch, 'total_epochs': epochs},
                   f'best_depth_model_{model_name}.pth')

# Load the best model
best_model = model.to(device)
best_model.load_state_dict(torch.load(f'best_depth_model_{model_name}.pth'))

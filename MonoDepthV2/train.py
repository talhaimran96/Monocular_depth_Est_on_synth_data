import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
from tqdm import tqdm
import albumentations as A
from DepthDataset import Depth_dataset  # Make sure to replace this with the correct path to your dataset class
import os
from torchvision.models import resnet18


class ResnetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetEncoder, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x1, x2, x3, x4, x5

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(DepthDecoder, self).__init__()
        self.upconv4 = nn.ConvTranspose2d(num_ch_enc[3], 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        x = self.upconv4(x5)
        x = self.upconv3(x + x4)
        x = self.upconv2(x + x3)
        x = self.upconv1(x + x2)
        return self.sigmoid(x)

resume_training = False

wandb.login(key='5acc1a4862a1e441f2368a1b8c979817fca97071')

dataset_path = "D:/Thesis/Dataset/resized_Dataset/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current Device :{device}")

# Define Hyperparameters
batch_size = 12
epochs = 12
starting_epoch = 0
initial_learning_rate = 0.001
random_seed = 42

# Initialize the model
encoder = ResnetEncoder(pretrained=True).to(device)
decoder = DepthDecoder(num_ch_enc=[64, 128, 256, 512]).to(device)

# Augmentations
augmentations = A.Compose(
    [A.RandomBrightnessContrast(p=0.2), A.HorizontalFlip(p=0.5), A.GaussNoise(p=0.2)]
)

train_dataset = Depth_dataset(dataset_path, dataset='train', transformations=augmentations)
test_dataset = Depth_dataset(dataset_path, dataset='test', transformations=None)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=initial_learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
loss_function = nn.MSELoss()
final_activation = nn.Sigmoid()

if resume_training:
    checkpoints = torch.load('best_depth_model_MonoDepthV2.pth')
    encoder.load_state_dict(checkpoints['encoder_state_dict'])
    decoder.load_state_dict(checkpoints['decoder_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    starting_epoch = int(checkpoints['current_epoch'])
    epochs = int(checkpoints['total_epochs'])

run = wandb.init(
    project="SkyGuard",
    config={
        "Model": "MonoDepthV2",
        "Initial_learning_rate": initial_learning_rate,
        "Epochs": epochs,
        "Batch_size": batch_size,
        "Random Seed": random_seed,
        "optimizer": "Adam",
        "Validation Ratio": 0,
        "Training Dataset": len(train_dataset)
    }
)

def train(encoder, decoder, train_loader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        features = encoder(data)
        output = final_activation(decoder(features))
        loss = torch.sqrt(criterion(output, target))
        loss.backward()
        optimizer.step()
        wandb.log({"Train_Batch_Loss": loss})

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate(encoder, decoder, val_loader, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            features = encoder(data)
            output = final_activation(decoder(features))
            loss = torch.sqrt(criterion(output, target))
            wandb.log({"Validation_Batch_Loss": loss})
            total_loss += loss.item()

    return total_loss / len(val_loader)

if __name__ == '__main__':
    best_val_loss = float('inf')
    for epoch in range(starting_epoch, epochs):
        train_loss = train(encoder, decoder, train_dataloader, loss_function, optimizer, device)
        val_loss = validate(encoder, decoder, test_dataloader, loss_function, device)
        wandb.log({"Train_epoch_loss": train_loss, "Validation_epoch_loss": val_loss,
                   "Current_learning_rate": scheduler.get_lr()[0]})
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step()

        # Save the model if the validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'current_epoch': epoch,
                'total_epochs': epochs
            }, 'best_depth_model_MonoDepthV2.pth')

    # Load the best model
    best_model_encoder = ResnetEncoder(num_layers=18, pretrained=True).to(device)
    best_model_decoder = DepthDecoder(num_ch_enc=[64, 128, 256, 512]).to(device)
    checkpoints = torch.load('best_depth_model_MonoDepthV2.pth')
    best_model_encoder.load_state_dict(checkpoints['encoder_state_dict'])
    best_model_decoder.load_state_dict(checkpoints['decoder_state_dict'])

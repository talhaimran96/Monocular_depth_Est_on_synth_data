import monai.networks.nets.swin_unetr as swin
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
import numpy as np
import cv2

resume_training = True

wandb.login(key='5acc1a4862a1e441f2368a1b8c979817fca97071')

dataset_path = "D:/Thesis/Dataset/resized_Dataset/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current Device :{device}")
# define Hyperparameters here
# encoder_stage = 'mit_b2'
batch_size = 8
epochs = 12
starting_epoch = 0
inital_learing_rate = 0.0001
random_seed = 42
model = swin.SwinUNETR(img_size=(384, 640), in_channels=3, out_channels=1, use_checkpoint=False,
                       spatial_dims=2).to(
    device)
# Augmentations
Augmentations = A.Compose(
    [A.RandomBrightnessContrast(p=0.2), A.HorizontalFlip(p=0.5), A.GaussNoise(p=0.2)])

train_dataset = Depth_dataset(dataset_path, dataset='train', transformations=Augmentations)
test_dataset = Depth_dataset(dataset_path, dataset='test', transformations=None)
# validation_dataset = Depth_dataset(dataset_path, dataset='validation', transformations=None)

# validation_ratio = 0.2
# validation_length = int(validation_ratio * len(train_dataset))

# train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
#                                                                   lengths=[len(train_dataset) - validation_length,
#                                                                            validation_length])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# model = smp.Unet(encoder_name=encoder_stage, encoder_weights='imagenet', in_channels=3, classes=1,
#                           activation=None, encoder_depth=5).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=inital_learing_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
loss_function = nn.MSELoss()
final_activation = nn.Sigmoid()


if resume_training:
    checkpoints = torch.load('best_depth_model_SwinUNETR_Monai.pth')
    model.load_state_dict(checkpoints['model_state_dict'])
    optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoints['schedular_state_dict'])
    starting_epoch = int(checkpoints['current_epoch'])
    epochs = int(checkpoints['total_epochs'])
    # print((scheduler.get_lr()[0]))

run = wandb.init(
    project="SkyGuard",
    config={
        "Model": "SwinUNETR_Monai + Gradient_RMSE",
        "Inital_learning_rate": inital_learing_rate,
        "Epochs": epochs,
        "Batch_size": batch_size,
        "Random Seed": random_seed,
        "optimizer": "Adam",
        "Validation Ratio": 0,
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


# def validate(model, val_loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#
#     with torch.no_grad():
#         for data, target in tqdm(val_loader):
#             data, target = data.to(device), target.to(device)
#
#             output = final_activation(model(data))
#             loss = torch.sqrt(criterion(output, target))
#             wandb.log({"Validation_Batch_Loss": loss})
#             total_loss += loss.item()
#
#     return total_loss / len(val_loader)


def validate(model, val_loader, criterion, device, n_samples_to_save=5):
    model.eval()
    total_loss = 0.0

    # Create a folder to save the samples
    save_dir = 'validation_samples'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(val_loader)):
            data, target = data.to(device), target.to(device)

            output = final_activation(model(data))
            loss = torch.sqrt(criterion(output, target))
            wandb.log({"Validation_Batch_Loss": loss})
            total_loss += loss.item()

            # Save n random samples
            if i < n_samples_to_save:
                # Convert data, target, and output tensors to numpy arrays
                data_np = data.cpu().numpy()
                target_np = target.cpu().numpy()
                output_np = output.cpu().numpy()

                # Select a random index
                random_index = np.random.randint(len(data_np))

                # Save input, target, and output images
                input_image = data_np[random_index].transpose(1, 2, 0)  # Assuming data is in NCHW format
                target_image = target_np[random_index].squeeze()
                output_image = output_np[random_index].squeeze()

                input_filename = os.path.join(save_dir, f'input_{i}.png')
                target_filename = os.path.join(save_dir, f'target_{i}.png')
                output_filename = os.path.join(save_dir, f'output_{i}.png')

                # Save images using your preferred method, for example, using PIL
                cv2.imwrite(input_filename, input_image)
                cv2.imwrite(target_filename, target_image)
                cv2.imwrite(output_filename, output_image)

    return total_loss / len(val_loader)


if __name__ == '__main__':
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
                       'best_depth_model_SwinUNETR_Monai.pth')

    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'schedular_state_dict': scheduler.state_dict(), 'current_epoch': epoch, 'total_epochs': epochs},
               'best_depth_model_SwinUNETR_Monai_final.pth')
    # Load the best model
    best_model = model.to(device)
    best_model.load_state_dict(torch.load('best_depth_model_SwinUNETR_Monai.pth'))
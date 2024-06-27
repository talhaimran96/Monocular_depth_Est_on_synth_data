import os
import monai.networks.nets.swin_unetr as swin
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from PIL import Image
# import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from DepthDataset import Depth_dataset

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the best model
model_path = 'best_depth_model_SwinUNETR_Monai.pth'

# Load the model
# model = smp.Unet(encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=1,
#                  activation=None, encoder_depth=5).to(device)
model = swin.SwinUNETR(img_size=(384, 1280 // 2), in_channels=3, out_channels=1, use_checkpoint=True,
                       spatial_dims=2).to(
    device)
checkpoints = torch.load(model_path)
model.load_state_dict(checkpoints['model_state_dict'])
model.eval()

# Create the test dataset and dataloader
dataset_path = "D:/Thesis/Dataset/resized_Dataset/"
test_dataset = Depth_dataset(dataset_path, train=False, transformations=None)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Output directory for test results
output_directory = "./test_results/SwinUnet/"
os.makedirs(output_directory, exist_ok=True)
final_activation = nn.Sigmoid()

loss_function = nn.MSELoss()

# Initialize total loss
total_loss = 0.0

# Inference and visualization loop
for i, (data, gt) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    data = data.to(device)
    gt = gt.to(device)

    output = final_activation(model(data))

    # Compute loss
    loss = loss_function(output, gt)
    total_loss += loss.item()

    # Post-process the output as needed (e.g., convert to image)
    output_image = transforms.ToPILImage()(output.cpu().squeeze(0))

    # Save the result as JPG
    image_name = f"result_{i + 1}.jpg"
    image_path = os.path.join(output_directory, image_name)
    output_image.save(image_path)

# Compute average loss
average_loss = total_loss / len(test_dataloader)

print(f"Inference and visualization completed. Results saved in: {output_directory}")
print(f"Average Loss over the test set: {average_loss}")

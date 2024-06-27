import os
from segmentation_models_pytorch import DeepLabV3Plus
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
model_path = 'best_depth_model_SMP_DEEPLabV3Plus_Gradient_loss_Final.pth'
encoder_stage = 'resnet34'

# Load the model
model = DeepLabV3Plus(encoder_name=encoder_stage, encoder_weights='imagenet', in_channels=3, classes=1,
                      activation='sigmoid', encoder_depth=5,
                      decoder_atrous_rates=(12, 24, 36)).to(device)
checkpoints = torch.load(model_path)
model.load_state_dict(checkpoints['model_state_dict'])
model.eval()

# Create the test dataset and dataloader
dataset_path = "D:/Thesis/Dataset/resized_Dataset/"
test_dataset = Depth_dataset(dataset_path, train=False, transformations=None)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Output directory for test results
output_directory = "./test_results/best_depth_model_SMP_DEEPLabV3Plus/"
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

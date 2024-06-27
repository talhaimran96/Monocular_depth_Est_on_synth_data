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
import cv2
from torchvision import transforms as T

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the best model
model_path = 'best_depth_model_SMP_DEEPLabV3Plus_standard_rmse.pth'
encoder_stage = 'resnet18'
# Load the model
# model = smp.Unet(encoder_name='mit_b0', encoder_weights=None, in_channels=3, classes=1,
#                  activation=None, encoder_depth=5).to(device)
model = DeepLabV3Plus(encoder_name=encoder_stage, encoder_weights='imagenet', in_channels=3, classes=1,
                      activation='sigmoid', encoder_depth=5,
                      decoder_atrous_rates=(12, 24, 36)).to(
    device)
checkpoints = torch.load('best_depth_model_SMP_DEEPLabV3Plus_standard_rmse.pth')
model.load_state_dict(checkpoints['model_state_dict'])
model.eval()

# Video input path
video_path = '../nust_yt_video.mp4'
cap = cv2.VideoCapture(video_path)

# Output directory for video results
output_directory = "./test_results/nust_yt_video/"
os.makedirs(output_directory, exist_ok=True)
final_activation = torch.nn.Sigmoid()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

standard_transformations = T.Compose(
    [T.ToTensor(), T.Resize((384, 1280 // 2)), T.Normalize(mean, std)])
# Inference and visualization loop for video frames
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame if needed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    # plt.subplot(1, 2, 1)
    # plt.imshow(frame)
    height, width, _ = frame.shape

    # Define the target size for center cropping
    target_height = 384
    target_width = 1280 // 2

    # Calculate cropping boundaries
    top = (height - target_height) // 2
    bottom = top + target_height
    left = (width - target_width) // 2
    right = left + target_width

    # Perform the center crop
    center_cropped_frame = frame[top:bottom, left:right, :]

    frame = standard_transformations(frame).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = (model(frame))

        # Post-process the output as needed (e.g., convert to image)
        output_image = transforms.ToPILImage()(output.cpu().squeeze(0))
    image_name = f"result_frame_{i + 1}.jpg"
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(output.cpu().squeeze(0)[0, :, :], cmap='plasma')
    # plt.savefig(os.path.join(output_directory, image_name))
    # Save the result as JPG

    image_path = os.path.join(output_directory, image_name)
    output_image.save(image_path)


# Release video capture object
cap.release()

print(f"Inference on video frames completed. Results saved in: {output_directory}")

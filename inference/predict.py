import os
import sys
import torch
import gdown
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import segmentation_models_pytorch as smp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pytorch_attention.attention_mechanisms.cbam import CBAM


class DeepLabV3PlusWithLayerWiseCBAM(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3PlusWithLayerWiseCBAM, self).__init__()
        self.model = smp.DeepLabV3Plus(encoder_name = "resnet50", encoder_weights = "imagenet", classes = num_classes, activation = None)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        
        # Initialize CBAM modules for each encoder stage with correct channel sizes
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        self.cbam5 = CBAM(2048)

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Get features from encoder
        features = self.encoder(x)
        
        # Apply CBAM to specific feature maps
        features[1] = self.cbam1(features[1])
        features[2] = self.cbam2(features[2])
        features[3] = self.cbam3(features[3])
        features[4] = self.cbam4(features[4])
        features[5] = self.cbam5(features[5])
        
        decoder_output = self.decoder(*features)
        x = self.segmentation_head(decoder_output)
        x = nn.functional.interpolate(x, size = input_size, mode = "bilinear", align_corners = False)
        
        return x


# Dictionary mapping checkpoint filenames to their Google Drive URLs
checkpoints = {
    "best_breast_uls_v3plusCBAMLayerWiseAtt_model.pth": "https://drive.google.com/uc?export=download&id=1099Oa67oJb1ngUp2V0uGNYp0mwLthh_W",
}

def download_checkpoint(checkpoint_name, destination_folder = "../ModelTrainingCheckpoint"):
    if checkpoint_name not in checkpoints:
        raise ValueError(f"Checkpoint '{checkpoint_name}' not found in the available URLs.")
    
    url = checkpoints[checkpoint_name]
    os.makedirs(destination_folder, exist_ok=True)
    destination_path = os.path.join(destination_folder, checkpoint_name)
    
    if not os.path.exists(destination_path):
        print(f"Downloading checkpoint '{checkpoint_name}'...")
        try:
            gdown.download(url, destination_path, quiet = False)
            print(f"Checkpoint downloaded and saved to '{destination_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoint: {e}")
    else:
        print(f"Checkpoint '{checkpoint_name}' already exists at '{destination_path}'")
    
    return destination_path

def load_model(model_class, checkpoint_path, device, *args, **kwargs):
    # Instantiate the model with provided arguments
    model = model_class(*args, **kwargs)
    model.to(device)
    model.eval()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")
    
    # Load the state dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle cases where the checkpoint might contain more than the state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # If the model was trained using DataParallel, adjust the keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded from '{checkpoint_path}'")
    
    return model

def preprocess_image(image_path, image_size=(512, 512)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at '{image_path}'")
    
    # Define the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def overlay_mask(original_image, mask, alpha=0.5):
    mask = mask.convert("L")
    mask_colored = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    mask_colored.paste(mask, (0, 0), mask)

    # Create a new image for the blended output
    combined = Image.new("RGBA", original_image.size)

    # Blend the images
    for x in range(original_image.width):
        for y in range(original_image.height):
            original_pixel = original_image.getpixel((x, y))
            mask_pixel = mask_colored.getpixel((x, y))

            # Calculate new pixel values using alpha blending
            new_pixel = (
                int(original_pixel[0] * (1 - alpha) + mask_pixel[0] * alpha),
                int(original_pixel[1] * (1 - alpha) + mask_pixel[1] * alpha),
                int(original_pixel[2] * (1 - alpha) + mask_pixel[2] * alpha),
                255  # Fully opaque
            )
            combined.putpixel((x, y), new_pixel)

    return combined.convert("RGB")  # Convert back to RGB for displaying


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def infer(image_path, checkpoint_path = "../ModelTrainingCheckpoint/best_breast_uls_v3plusCBAMLayerWiseAtt_model.pth", model_class = DeepLabV3PlusWithLayerWiseCBAM, device = DEVICE, num_classes = 1, pred_threshold = 0.5):
    # Load and preprocess the image
    image = preprocess_image(image_path)
    original_image = Image.open(image_path).convert("RGB")  # Load the original image for overlay
    image = image.to(device)
    
    # Load the model with the required number of classes
    model = load_model(model_class, checkpoint_path, device, num_classes=num_classes)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        output = nn.functional.sigmoid(output)
        output = output.squeeze(0).cpu()
        if output.shape[0] == 1:
            output = output.squeeze(0)
        else:
            output = torch.argmax(output, dim=0).float()
        
        output = (output > pred_threshold).float().numpy()  # Binarize
    
    # Convert the output to PIL Image
    prediction = Image.fromarray((output * 255).astype("uint8"))

    # Overlay the mask on the original image
    overlayed_output = overlay_mask(original_image, prediction)

    original_image.save("image.jpg")
    prediction.save("output.jpg")
    overlayed_output.save("overlay_output.jpg")

    # return (image, output, overlayed_output)

def visualize_inference(overlayed_output):
    plt.figure(figsize=(10, 5))
    
    # Overlayed Image
    plt.imshow(overlayed_output)
    plt.title("Overlayed Output")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
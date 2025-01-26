import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


class Dinov2:

    def __init__(self):
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14")
        self.model.eval()
        self.model.cuda()

        # Define the transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225)
            )
        ])

        self.patch_size = self.model.patch_size

    def prepare_image(self, image):
        # input image is RGB and numpy

        # Convert to PIL
        image = Image.fromarray(image)
        image_tensor = self.transforms(image)

        # crop image to dimensions that are multiples of patch size
        height, width = image_tensor.shape[1:]
        cropped_height = height - height % self.patch_size
        cropped_width = width - width % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size
    
    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            features = self.model.get_intermediate_layers(image_batch)[0].squeeze()

        return features.cpu().numpy()
    
    def idx_to_pos(self, idx, grid_size):
        row = (idx // grid_size[1])*self.model.patch_size + (self.model.patch_size/2)
        col = (idx % grid_size[1])*self.model.patch_size + (self.model.patch_size/2)


if __name__ == "__main__":
    object = Dinov2()
    
    print("patch size", object.patch_size)

    image = cv2.imread("/home/niru/Downloads/asd1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread("/home/niru/Downloads/asd2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image_tensor, grid_size = object.prepare_image(image)
    image_tensor2, grid_size2 = object.prepare_image(image2)

    print("image tensor shape", image_tensor.shape)
    print("grid size", grid_size)

    features = object.extract_features(image_tensor)

    print("features shape", features.shape)

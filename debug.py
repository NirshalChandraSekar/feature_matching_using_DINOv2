import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


class Dinov2:

    def __init__(self):
        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device", self.device)
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
    
    def idx_to_source_position(self, idx, grid_size, patch_size):
        row = (idx // grid_size[1])*patch_size + patch_size / 2
        col = (idx % grid_size[1])*patch_size + patch_size / 2
        return row, col
    
    def pixel_to_idx(self, pixel, grid_size, patch_size):
        row, col = pixel
        idx = int(row // patch_size * grid_size[1] + col // patch_size)
        return idx
    
    def knn_matcher(self, features1, features2, k=1):
        # Fit a KNN model
        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(features2)
        distances, indices = knn.kneighbors(features1)
        return distances, indices

    def get_best_matches_with_indices(self, distances, match1to2, n):
        """
        Get the best `n` matches based on the closest distance, along with their corresponding indices in features1 and features2.

        Args:
            distances (np.array): Array of distances for each match, shape (N, 1).
            match1to2 (np.array): Array of matched indices, shape (N, 1).
            n (int): Number of best matches to return.

        Returns:
            best_distances (np.array): Distances of the best `n` matches.
            best_indices_features1 (np.array): Indices in features1 for the best `n` matches.
            best_indices_features2 (np.array): Indices in features2 for the best `n` matches.
        """
        # Flatten the distances and match arrays
        distances = distances.flatten()
        match1to2 = match1to2.flatten()

        # Create an array of indices for features1
        indices_features1 = np.arange(len(distances))

        # Sort by distance (ascending order)
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        sorted_matches_features2 = match1to2[sorted_indices]
        sorted_indices_features1 = indices_features1[sorted_indices]

        # Select the top `n` matches
        best_distances = sorted_distances[:n]
        best_indices_features1 = sorted_indices_features1[:n]
        best_indices_features2 = sorted_matches_features2[:n]

        return best_distances, best_indices_features1, best_indices_features2

        
def visualize_matches_with_connection_patches(image1, image2, best_indices_features1, best_indices_features2, grid_size1, grid_size2, patch_size):
    """
    Visualize the best matches between two images using ConnectionPatch.

    Args:
        image1 (np.array): First image (RGB).
        image2 (np.array): Second image (RGB).
        best_indices_features1 (np.array): Indices in features1 for the best matches.
        best_indices_features2 (np.array): Indices in features2 for the best matches.
        grid_size1 (tuple): Grid size of image1 (height, width).
        grid_size2 (tuple): Grid size of image2 (height, width).
        patch_size (int): Patch size used in the model.
    """
    # Create a figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Display the images
    ax1.imshow(image1)
    ax2.imshow(image2)

    # Draw lines for the best matches
    for idx1, idx2 in zip(best_indices_features1, best_indices_features2):
        # Get the positions of the patches
        x_center1, y_center1 = object.idx_to_source_position(idx1, grid_size1, patch_size)
        x_center2, y_center2 = object.idx_to_source_position(idx2, grid_size2, patch_size)

        # Create a ConnectionPatch
        con = ConnectionPatch(
            xyA=(y_center2, x_center2),  # Point in image2
            xyB=(y_center1, x_center1),  # Point in image1
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            color=np.random.rand(3,),  # Random color for each line
            linewidth=1,
        )
        ax2.add_artist(con)

    # Set titles
    ax1.set_title("Image 1")
    ax2.set_title("Image 2")

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    object = Dinov2()
    
    print("patch size", object.patch_size)

    image1 = cv2.imread("images/van1.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread("images/van2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image_tensor1, grid_size1 = object.prepare_image(image1)
    image_tensor2, grid_size2 = object.prepare_image(image2)

    print("image tensor 1 shape", image_tensor1.shape)
    print("grid size 1", grid_size1)

    print("image tensor 2 shape", image_tensor2.shape)
    print("grid size 2", grid_size2)

    feature1 = object.extract_features(image_tensor1)
    feature2 = object.extract_features(image_tensor2)

    print("feature 1 shape", feature1.shape)
    print("feature 2 shape", feature2.shape)

    distances, match1to2 = object.knn_matcher(feature1, feature2, k=1)
    match1to2 = np.array(match1to2)
    
    n = 100  # Number of best matches to retrieve
    best_distances, best_indices_features1, best_indices_features2 = object.get_best_matches_with_indices(distances, match1to2, n)

    visualize_matches_with_connection_patches(image1, image2, best_indices_features1, best_indices_features2, grid_size1, grid_size2, object.patch_size)




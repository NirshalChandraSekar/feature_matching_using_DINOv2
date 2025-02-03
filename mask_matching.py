from debug import Dinov2, visualize_matches_with_connection_patches
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device", device)

else:
    device = torch.device("cpu")
    print("Device", device)



class SAM:
    def __init__(self):
        checkpoint = "/home/nirshal/codes/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        self.model = build_sam2(model_cfg, checkpoint, device = device)
        self.predictor = SAM2ImagePredictor(self.model)

    def get_manual_point(self, image):
        point = []
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("reference image", image)
                point.append([x, y]) # row, col

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("reference image", image)
        cv2.setMouseCallback("reference image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return point # row, col
    
    def sement_using_point(self, image, point):
        self.predictor.set_image(image)
        point = np.array(point)
        label = np.array([1]*len(point))

        masks, _, _ = self.predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=False,
        )

        return masks

class MaskMatch:
    def __init__(self):
        pass



if __name__ == "__main__":
    segmentor = SAM()
    dino = Dinov2()

    image1 = cv2.imread("images/a1.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread("images/a2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    point = segmentor.get_manual_point(image1)
    point = np.array(point)
    masks = segmentor.sement_using_point(image1, point)
    masks = np.transpose(masks, (1, 2, 0))

    cv2.imshow("mask", masks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_tensor1, grid_size1 = dino.prepare_image(image1)
    masks = masks[:image_tensor1.shape[1], :image_tensor1.shape[2], :]
    image1 = image1[:image_tensor1.shape[1], :image_tensor1.shape[2], :]
    
    image_tensor2, grid_size2 = dino.prepare_image(image2)
    image2 = image2[:image_tensor2.shape[1], :image_tensor2.shape[2], :]

    print("grid size 1", grid_size1)
    print("grid size 2", grid_size2)

    feature1 = dino.extract_features(image_tensor1)
    feature1 = np.reshape(feature1, (grid_size1[0], grid_size1[1], -1))
    feature2 = dino.extract_features(image_tensor2)
    feature2 = np.reshape(feature2, (grid_size2[0], grid_size2[1], -1))

    print("feature 1 shape", feature1.shape)
    print("feature 2 shape", feature2.shape)

    

    feature1_flat = feature1.reshape(-1, feature1.shape[-1])  # Shape: (H_grid * W_grid, feature_dim)
    feature2_flat = feature2.reshape(-1, feature2.shape[-1])  # Shape: (H_grid * W_grid, feature_dim)

    # Perform KNN matching
    distances, match1to2 = dino.knn_matcher(feature1_flat, feature2_flat, k=1)


    filtered_indices_features1 = []
    filtered_indices_features2 = []
    filtered_distances = []

    for idx1, (dist, idx2) in enumerate(zip(distances, match1to2)):
        # Get patch center in image1
        row, col = dino.idx_to_source_position(idx1, grid_size1, dino.patch_size)
        row = int(row)
        col = int(col)

        # Check if the patch center is inside the SAM mask
        if masks[row, col, 0] == 1:  # Mask is 1 at this location
            filtered_indices_features1.append(idx1)
            filtered_indices_features2.append(idx2[0])  # idx2 is a list of matches (k=1)
            filtered_distances.append(dist[0])  # dist is a list of distances (k=1)

    # Convert lists to numpy arrays
    filtered_indices_features1 = np.array(filtered_indices_features1)
    filtered_indices_features2 = np.array(filtered_indices_features2)
    filtered_distances = np.array(filtered_distances)

    print("Number of matches within SAM mask:", len(filtered_indices_features1))

    # Sort the filtered matches by distance (ascending order)
    sorted_indices = np.argsort(filtered_distances)
    sorted_indices_features1 = filtered_indices_features1[sorted_indices]
    sorted_indices_features2 = filtered_indices_features2[sorted_indices]
    sorted_distances = filtered_distances[sorted_indices]

    # Select the top n matches
    n = 50  # Number of top matches to select
    top_indices_features1 = sorted_indices_features1[:n]
    top_indices_features2 = sorted_indices_features2[:n]

    print("Number of top matches to visualize:", len(top_indices_features1))

    # Visualize the top n filtered matches
    visualize_matches_with_connection_patches(
        image1, image2,
        top_indices_features1, top_indices_features2,
        grid_size1, grid_size2, dino.patch_size, object=dino
    )
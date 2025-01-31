from debug import Dinov2
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_manual_point(image):

    point = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("reference image", image)
            point.append([y, x])

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("reference image", image)
    cv2.setMouseCallback("reference image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return point # row, col

def compute_feature_distance(index_1, feature1, feature2, grid_size2):

    distances = np.linalg.norm(feature2 - feature1[index_1], axis=1)
    return distances

if __name__ == "__main__":
    obj = Dinov2()

    image1 = cv2.imread("images/bottle1.jpg")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread("images/bottle2.jpg")
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image_tensor1, grid_size1 = obj.prepare_image(image1)
    image_tensor2, grid_size2 = obj.prepare_image(image2)

    feature1 = obj.extract_features(image_tensor1)
    feature2 = obj.extract_features(image_tensor2)

    """
    Manually pick a point in first image using click event and print the point
    """

    point = get_manual_point(image1)

    point_idx = obj.pixel_to_idx(point[0], grid_size1, obj.patch_size)

    distances = compute_feature_distance(point_idx, feature1, feature2, grid_size2)

    distances = np.reshape(distances, (grid_size2[0], grid_size2[1]))
    interpolated_distances = cv2.resize(distances, (image_tensor2.shape[2], image_tensor2.shape[1]), interpolation=cv2.INTER_CUBIC)

    # change image_tensor2 to numpy array
    image_tensor2 = image_tensor2.numpy()
    image_tensor2 = np.transpose(image_tensor2, (1, 2, 0))

    #ovelay the distance map on the image
    plt.imshow(image2)
    plt.imshow(interpolated_distances, alpha=0.7)
    plt.colorbar()
    plt.show()


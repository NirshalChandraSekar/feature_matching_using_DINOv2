import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

import supervision as sv
from scipy.signal import find_peaks

import mediapipe as mp



class VideoSegmentation:
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', self.device)
        if self.device == 'cuda':
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        checkpoint = "/home/nirshal/codes/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        config = "configs/sam2.1/sam2.1_hiera_t.yaml"

        self.predictor = build_sam2_video_predictor(config, checkpoint, device = self.device)

    def split_video_frames(self, video_path):
        print("splitting the video frames")
        frames_generator = sv.get_video_frames_generator(video_path)
        
        # Sink to save the individual frames as JPEG files
        sink = sv.ImageSink(target_dir_path="frames",
                            image_name_pattern="{:05d}.jpeg")
        with sink:
            for frame in frames_generator:
                sink.save_image(frame)

        # Retrieve the first frame for mask generation
        frame_files = sorted(os.listdir("frames"))
        first_frame = cv2.imread(os.path.join("frames", frame_files[0]))

        return first_frame 
    
    def interactive_point_selection(self, frame):
        point = []
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("reference frame", frame)
                point.append([x, y]) # col, row

        cv2.imshow("reference frame", frame)
        cv2.setMouseCallback("reference frame", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        point = np.array(point, np.float32)
        labels = np.array([1]*len(point), np.int32)


        return point, labels # Point --> row, col
    
    def segment_video(self, video_path, point, labels):
        inference_state = self.predictor.init_state(video_path = video_path)
        self.predictor.reset_state(inference_state)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point,
            labels=labels,
        )

        mask = out_mask_logits[0].cpu().numpy().transpose(1, 2, 0)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            frame_data = {}  # Dictionary to store object data for the current frame
            for i, out_obj_id in enumerate(out_obj_ids):
                # Generate the mask for the current object
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 255

                # Find contours and compute bounding box
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contours = np.concatenate(contours)
                    x, y, w, h = cv2.boundingRect(contours)
                else:
                    x, y, w, h = 0, 0, 0, 0  # Default bounding box if no contours are found

                # Store the mask and bbox in the frame data
                frame_data[out_obj_id] = {
                    "mask": mask,
                    "bbox": [x + w//2, y + h//2, w, h]
                }
            # Add the frame data to the video_segments dictionary
            video_segments[out_frame_idx] = frame_data


        return video_segments

def find_pick_up_frame(video_segments):
    x_cord = []
    y_cord = []

    for key in video_segments:
        mask = video_segments[key][1]["mask"]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # draw the bounding mask on this mask image
        x, y, w, h = video_segments[key][1]['bbox']
        cv2.rectangle(mask, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), 2)      
        x_cord.append(x)
        y_cord.append(y) 
        # cv2.imshow("mask", mask)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break

    # cv2.destroyAllWindows()

    # plot the xcord and ycords seperately in a graph
    peaks, _ = find_peaks(y_cord, height=0)
    pick_up_frame = peaks[-1]

    # plt.plot(x_cord)
    # plt.plot(y_cord, label="y_cord")
    # plt.plot(pick_up_frame, y_cord[pick_up_frame], "x")

    # plt.show()

    return pick_up_frame
    
class HandTracking:
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the HandTracking class.
        
        :param static_image_mode: Whether to treat input as static images
        :param max_num_hands: Maximum number of hands to detect
        :param min_detection_confidence: Minimum confidence for hand detection
        :param min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def convert_flipped_to_original_coords(self, x, y, width):
        """
        Convert coordinates from flipped image back to original image coordinates.
        
        :param x: x-coordinate in flipped image
        :param y: y-coordinate in flipped image
        :param width: image width
        :return: tuple of (x, y) in original image
        """
        # Only x needs to be transformed since vertical flip wasn't performed
        original_x = width - x
        return (original_x, y)

    def track_hands(self, video_path=0, save_video=False, output_path=None):
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Results dictionary
        results_dict = {}
        frame_number = 0
        
        # Process video frames
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
                
            # Get original dimensions
            h, w, _ = img.shape
            
            # Flip image for display only
            display_img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Process frame
            hand_results = self.hands.process(img_rgb)
            
            # Initialize frame results
            frame_results = {
                'right_hand': {},
                'left_hand': {}
            }
            
            # Process detected hands
            if hand_results.multi_hand_landmarks:
                for hand_no, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # For flipped image, right hand appears as left and vice versa
                    detected_label = hand_results.multi_handedness[hand_no].classification[0].label
                    hand_type = 'left_hand' if detected_label == 'Right' else 'right_hand'
                    
                    # Extract thumb and index finger tip coordinates
                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Convert normalized coordinates to pixel coordinates in flipped image
                    thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Convert coordinates back to original image space
                    orig_thumb_x, orig_thumb_y = self.convert_flipped_to_original_coords(thumb_x, thumb_y, w)
                    orig_index_x, orig_index_y = self.convert_flipped_to_original_coords(index_x, index_y, w)
                    
                    # Store original image coordinates
                    frame_results[hand_type] = {
                        'thumbtip': (orig_thumb_x, orig_thumb_y),
                        'indextip': (orig_index_x, orig_index_y)
                    }
                    
                    # Draw landmarks on display image
                    self.mp_draw.draw_landmarks(
                        display_img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Store frame results
            results_dict[frame_number] = frame_results
            
            # Display frame (optional)
            # cv2.imshow("Hand Tracking", display_img)
            
            # Increment frame number
            frame_number += 1
            
            # Exit on 'q' key press
            # if cv2.waitKey(20) & 0xFF == ord('q'):
            #     break
        
        # Release resources
        cap.release()
        # cv2.destroyAllWindows()
        
        return results_dict


if __name__ == '__main__':
    
    vs = VideoSegmentation()
    # first_frame = vs.split_video_frames("videos/test.mp4")

    # point, labels = vs.interactive_point_selection(first_frame)
    # labels = np.array([1]*len(point), np.int32)

    # video_segments = vs.segment_video("frames", point, labels)

    # np.save("video_segments.npy", video_segments)

    video_segments = np.load("video_segments.npy", allow_pickle=True).item()

    pick_up_frame = find_pick_up_frame(video_segments)
    print(pick_up_frame)

    hand_tracker = HandTracking()
    results = hand_tracker.track_hands("videos/test.mp4")
    
    print(results[pick_up_frame])

    image = cv2.imread("frames/00000.jpeg")
   

    if 'thumbtip' in results[pick_up_frame]['left_hand']:
        cv2.circle(image, results[pick_up_frame]['left_hand']['thumbtip'], 5, (0, 255, 0), -1)
    if 'thumbtip' in results[pick_up_frame]['right_hand']:
        cv2.circle(image, results[pick_up_frame]['right_hand']['thumbtip'], 5, (0, 255, 0), -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
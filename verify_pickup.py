import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

import supervision as sv
from scipy.signal import find_peaks



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
        cv2.imshow("mask", mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # plot the xcord and ycords seperately in a graph
    peaks, _ = find_peaks(y_cord, height=0)
    pick_up_frame = peaks[-1]

    # plt.plot(x_cord)
    plt.plot(y_cord, label="y_cord")
    plt.plot(pick_up_frame, y_cord[pick_up_frame], "x")

    plt.show()

    return pick_up_frame
    
class HandTracking:
    def __init__(self):
        pass

if __name__ == '__main__':
    
    vs = VideoSegmentation()
    # first_frame = vs.split_video_frames("videos/test.mp4")

    # point, labels = vs.interactive_point_selection(first_frame)
    # labels = np.array([1]*len(point), np.int32)

    # video_segments = vs.segment_video("frames", point, labels)

    # np.save("video_segments.npy", video_segments)

    video_segments = np.load("video_segments.npy", allow_pickle=True).item()

    pick_up_frame = find_pick_up_frame(video_segments)

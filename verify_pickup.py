import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor

import supervision as sv


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

        return point # row, col
    


if __name__ == '__main__':
    
    vs = VideoSegmentation()
    first_frame = vs.split_video_frames("test.mp4")
    inference_state = vs.predictor.init_state(video_path = "frames")
    vs.predictor.reset_state(inference_state)

    point = vs.interactive_point_selection(first_frame)
    point = np.array(point, np.float32)
    labels = np.array([1]*len(point), np.int32)

    _, out_obj_ids, out_mask_logits = vs.predictor.add_new_points_or_box(
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
    for out_frame_idx, out_obj_ids, out_mask_logits in vs.predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().transpose(1, 2, 0)
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print(video_segments[0][1].shape)
    for key in video_segments:
        mask = video_segments[key][1]
        mask = (mask.astype(np.uint8)) * 255

        cv2.imshow("mask", mask)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
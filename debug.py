import cv2
import mediapipe as mp
import time

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

    def track_hands(self, video_path=0, save_video=False, output_path=None):
        """
        Track hands in a video and return their landmark coordinates.
        
        :param video_path: Path to video file or camera index (0 for webcam)
        :param save_video: Whether to save the output video
        :param output_path: Path to save the output video
        :return: Dictionary of hand landmarks for each frame
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Video writer setup if saving video
        if save_video and output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                20, 
                (frame_width, frame_height)
            )
        
        # Results dictionary
        results_dict = {}
        frame_number = 0
        
        # Process video frames
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            
            # Flip and convert image
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
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
                    # Determine hand orientation (left or right)
                    hand_type = 'right_hand' if hand_results.multi_handedness[hand_no].classification[0].label == 'Right' else 'left_hand'
                    
                    # Get image dimensions
                    h, w, _ = img.shape
                    
                    # Extract thumb and index finger tip coordinates
                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Store pixel coordinates
                    frame_results[hand_type] = {
                        'thumbtip': (int(thumb_tip.x * w), int(thumb_tip.y * h)),
                        'indextip': (int(index_tip.x * w), int(index_tip.y * h))
                    }
                    
                    # Optionally draw landmarks
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Store frame results
            results_dict[frame_number] = frame_results
            
            # Save frame to output video if enabled
            if save_video and output_path:
                out.write(img)
            
            # Display frame (optional)
            cv2.imshow("Hand Tracking", img)
            
            # Increment frame number
            frame_number += 1
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if save_video and output_path:
            out.release()
        cv2.destroyAllWindows()
        
        return results_dict

# Usage example
if __name__ == "__main__":
    # Create HandTracking instance
    hand_tracker = HandTracking()
    
    # Track hands from webcam
    results = hand_tracker.track_hands()
    
    # Print results (optional)
    for frame, hands in results.items():
        print(f"Frame {frame}:")
        print("Right Hand:", hands['right_hand'])
        print("Left Hand:", hands['left_hand'])
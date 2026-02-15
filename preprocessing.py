import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Optional, List, Tuple
import pickle
from tqdm import tqdm



class PhoenixKeypointExtractor:
    """Enhanced keypoint extractor with interpolation and normalization"""
    
    # MediaPipe to PoseNet landmark mapping (keeping only relevant landmarks)
    POSE_LANDMARKS_TO_KEEP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
    
    def __init__(self, model_complexity: int = 1):
        self.model_complexity = model_complexity
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=model_complexity
        )
        
    def extract_from_video(self, video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """Extract keypoints from video file"""
        cap = cv2.VideoCapture(video_path)
        pose_data, left_hand_data, right_hand_data = [], [], []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Flip for selfie view and convert to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            
            # Extract pose landmarks
            if results.pose_landmarks:
                pose_data.append([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
            else:
                pose_data.append([(0.0, 0.0, 0.0)] * 33)
                
            # Extract left hand landmarks
            if results.left_hand_landmarks:
                left_hand_data.append([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
            else:
                left_hand_data.append([(0.0, 0.0, 0.0)] * 21)
                
            # Extract right hand landmarks
            if results.right_hand_landmarks:
                right_hand_data.append([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])
            else:
                right_hand_data.append([(0.0, 0.0, 0.0)] * 21)
                
        cap.release()
        
        # Process the extracted data
        return self._process_keypoints(pose_data, left_hand_data, right_hand_data, max_frames)
    
    def extract_from_frames(self, frame_dir: Path, max_frames: Optional[int] = None) -> np.ndarray:
        """Extract keypoints from directory of frames (PHOENIX format)"""
        frame_files = sorted(frame_dir.glob("*.png"), key=lambda x: int(x.stem.replace('images', '')))
        
        pose_data, left_hand_data, right_hand_data = [], [], []
        
        for frame_path in frame_files[:max_frames] if max_frames else frame_files:
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"Warning: Could not read {frame_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            
            # Extract landmarks
            if results.pose_landmarks:
                pose_data.append([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
            else:
                pose_data.append([(0.0, 0.0, 0.0)] * 33)
                
            if results.left_hand_landmarks:
                left_hand_data.append([(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark])
            else:
                left_hand_data.append([(0.0, 0.0, 0.0)] * 21)
                
            if results.right_hand_landmarks:
                right_hand_data.append([(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark])
            else:
                right_hand_data.append([(0.0, 0.0, 0.0)] * 21)
        
        if not pose_data:  # No frames processed
            print(f"Warning: No frames found in {frame_dir}")
            # Return zeros array with correct shape
            return np.zeros((max_frames or 1, 225))
        
        return self._process_keypoints(pose_data, left_hand_data, right_hand_data, max_frames)
    
    def _process_keypoints(self, pose_data, left_hand_data, right_hand_data, max_frames):
        """Apply interpolation, normalization, and feature engineering"""
        
        # Convert to numpy arrays first to check shapes
        pose_array = np.array(pose_data)
        left_hand_array = np.array(left_hand_data)
        right_hand_array = np.array(right_hand_data)
        
        print(f"Debug - Shapes before processing: pose={pose_array.shape}, left={left_hand_array.shape}, right={right_hand_array.shape}")
        
        # Step 1: Interpolate missing frames
        pose_data = self._interpolate_keypoints(pose_data)
        left_hand_data = self._interpolate_keypoints(left_hand_data)
        right_hand_data = self._interpolate_keypoints(right_hand_data)
        
        # Step 2: Put hands in body coordinates
        pose_data, left_hand_data, right_hand_data = self._put_hands_in_body(
            pose_data, left_hand_data, right_hand_data
        )
        
        # Step 3: Convert to PoseNet format (keep only relevant landmarks)
        pose_data = self._mediapipe_to_posenet(pose_data)
        
        # Step 4: Apply chicken neck normalization
        pose_data, left_hand_data, right_hand_data = self._chicken_neck_normalization(
            pose_data, left_hand_data, right_hand_data
        )
        
        # Convert to numpy arrays
        pose_array = np.array(pose_data)
        left_hand_array = np.array(left_hand_data)
        right_hand_array = np.array(right_hand_data)
        
        print(f"Debug - Shapes after processing: pose={pose_array.shape}, left={left_hand_array.shape}, right={right_hand_array.shape}")
        
        # Ensure all arrays have the same number of frames
        min_frames = min(pose_array.shape[0], left_hand_array.shape[0], right_hand_array.shape[0])
        pose_array = pose_array[:min_frames]
        left_hand_array = left_hand_array[:min_frames]
        right_hand_array = right_hand_array[:min_frames]
        
        # Reshape to 2D if needed (frames, features)
        pose_reshaped = pose_array.reshape(min_frames, -1)
        left_reshaped = left_hand_array.reshape(min_frames, -1)
        right_reshaped = right_hand_array.reshape(min_frames, -1)
        
        print(f"Debug - Shapes after reshape: pose={pose_reshaped.shape}, left={left_reshaped.shape}, right={right_reshaped.shape}")
        
        # Step 5: Concatenate all features
        data = np.concatenate([pose_reshaped, left_reshaped, right_reshaped], axis=1)
        
        print(f"Debug - Concatenated shape: {data.shape}")
        
        # Step 6: Pad or truncate to max_frames
        if max_frames:
            if len(data) < max_frames:
                padding = max_frames - len(data)
                # Create padding array with same number of features
                padding_array = np.zeros((padding, data.shape[1]))
                data = np.vstack([data, padding_array])
                print(f"Debug - After padding: {data.shape}")
            else:
                data = data[:max_frames]
                print(f"Debug - After truncation: {data.shape}")
        
        # Final check for NaN values
        if np.any(np.isnan(data)):
            print("Warning: NaN values detected, replacing with 0")
            data = np.nan_to_num(data)
        
        return data
    
    def _interpolate_keypoints(self, data: List) -> List:
        """Interpolate missing keypoints using neighboring frames"""
        if not data:
            return data
            
        data_array = np.array(data, dtype=np.float32)
        
        for i in range(len(data_array)):
            if np.all(data_array[i] == 0):
                # Find last valid frame
                last_valid = i - 1
                while last_valid >= 0 and np.all(data_array[last_valid] == 0):
                    last_valid -= 1
                    
                # Find next valid frame
                next_valid = i + 1
                while next_valid < len(data_array) and np.all(data_array[next_valid] == 0):
                    next_valid += 1
                    
                if last_valid >= 0 and next_valid < len(data_array):
                    # Linear interpolation
                    alpha = (i - last_valid) / (next_valid - last_valid)
                    data_array[i] = (1 - alpha) * data_array[last_valid] + alpha * data_array[next_valid]
                elif last_valid >= 0:
                    data_array[i] = data_array[last_valid]
                elif next_valid < len(data_array):
                    data_array[i] = data_array[next_valid]
                    
        return data_array.tolist()
    
    def _put_hands_in_body(self, pose, left_hand, right_hand):
        """Transform hand coordinates relative to wrists"""
        pose_array = np.array(pose, dtype=np.float32)
        left_hand_array = np.array(left_hand, dtype=np.float32)
        right_hand_array = np.array(right_hand, dtype=np.float32)
        
        # Ensure we have at least 17 landmarks (MediaPipe pose has 33, but we need index 15,16)
        if pose_array.shape[1] < 17:
            print(f"Warning: Pose has only {pose_array.shape[1]} landmarks, expected at least 17")
            return pose.tolist(), left_hand.tolist(), right_hand.tolist()
        
        for frame in range(len(pose_array)):
            # Left wrist is index 15, right wrist is index 16 in MediaPipe
            left_wrist = pose_array[frame, 15].copy()
            right_wrist = pose_array[frame, 16].copy()
            
            # Transform left hand relative to left wrist
            if frame < len(left_hand_array) and not np.all(left_hand_array[frame] == 0):
                left_hand_array[frame] = left_hand_array[frame] - left_wrist
                
            # Transform right hand relative to right wrist
            if frame < len(right_hand_array) and not np.all(right_hand_array[frame] == 0):
                right_hand_array[frame] = right_hand_array[frame] - right_wrist
                
        return pose_array.tolist(), left_hand_array.tolist(), right_hand_array.tolist()
    
    def _mediapipe_to_posenet(self, pose):
        """Keep only relevant pose landmarks (similar to PoseNet format)"""
        pose_array = np.array(pose, dtype=np.float32)
        
        # Check if we have enough landmarks
        max_idx = max(self.POSE_LANDMARKS_TO_KEEP)
        if pose_array.shape[1] <= max_idx:
            print(f"Warning: Pose has {pose_array.shape[1]} landmarks, but need at least {max_idx+1}")
            # Pad with zeros if needed
            if pose_array.shape[1] < max_idx + 1:
                padding = np.zeros((pose_array.shape[0], max_idx + 1 - pose_array.shape[1], 3))
                pose_array = np.concatenate([pose_array, padding], axis=1)
        
        # Select only the landmarks we want to keep
        selected = pose_array[:, self.POSE_LANDMARKS_TO_KEEP]
        return selected.tolist()
    
    def _chicken_neck_normalization(self, pose, left_hand, right_hand):
        """Normalize around the neck point"""
        pose_array = np.array(pose, dtype=np.float32)
        left_hand_array = np.array(left_hand, dtype=np.float32)
        right_hand_array = np.array(right_hand, dtype=np.float32)
        
        for frame in range(len(pose_array)):
            # Neck is average of shoulders (indices 1 and 2 in selected landmarks)
            if pose_array.shape[1] > 2:
                neck = (pose_array[frame, 1] + pose_array[frame, 2]) * 0.5
                
                # Translate everything so neck is at origin
                pose_array[frame] = pose_array[frame] - neck
                
                if frame < len(left_hand_array) and not np.all(left_hand_array[frame] == 0):
                    left_hand_array[frame] = left_hand_array[frame] - neck
                    
                if frame < len(right_hand_array) and not np.all(right_hand_array[frame] == 0):
                    right_hand_array[frame] = right_hand_array[frame] - neck
                    
        return pose_array.tolist(), left_hand_array.tolist(), right_hand_array.tolist()
    
    def close(self):
        self.holistic.close()
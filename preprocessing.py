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
    """
    Enhanced keypoint extractor with interpolation and normalization.
    
    Landmark budget (75 joints × 3 coords = 225 features):
        Pose  : 13 selected MediaPipe landmarks  → 13 × 3 =  39
        L Hand: 21 landmarks                     → 21 × 3 =  63
        R Hand: 21 landmarks                     → 21 × 3 =  63
        Face  : 20 selected landmarks             → 20 × 3 =  60
        ─────────────────────────────────────────────────────────
        Total : 75 joints                         → 75 × 3 = 225
    """
    
    # MediaPipe pose → keep upper body + hips (same as before)
    POSE_LANDMARKS_TO_KEEP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
    
    # MediaPipe FaceMesh indices for sign-language-relevant regions.
    # 20 landmarks covering: lips (8), eyebrows (6), eyes (4), nose (2)
    #
    # Lips outer:   61 (left corner), 291 (right corner),
    #               0 (top center), 17 (bottom center),
    #               39 (upper-left), 269 (upper-right),
    #               181 (lower-left), 405 (lower-right)
    # Left eyebrow: 70 (inner), 63 (mid), 105 (outer)
    # Right eyebrow:300 (inner), 293 (mid), 334 (outer)
    # Left eye:     33 (outer corner), 133 (inner corner)
    # Right eye:    263 (outer corner), 362 (inner corner)
    # Nose:         1 (tip), 2 (bridge)
    FACE_LANDMARKS_TO_KEEP = [
        # Lips (8)
        61, 291, 0, 17, 39, 269, 181, 405,
        # Left eyebrow (3)
        70, 63, 105,
        # Right eyebrow (3)
        300, 293, 334,
        # Left eye (2)
        33, 133,
        # Right eye (2)
        263, 362,
        # Nose (2)
        1, 2,
    ]
    
    def __init__(self, model_complexity: int = 1):
        self.model_complexity = model_complexity
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=model_complexity
        )
        
        # Verify landmark budget
        self._num_pose = len(self.POSE_LANDMARKS_TO_KEEP)   # 13
        self._num_hand = 21                                   # per hand
        self._num_face = len(self.FACE_LANDMARKS_TO_KEEP)    # 20
        self._total_joints = self._num_pose + 2 * self._num_hand + self._num_face  # 75
        self._feature_dim = self._total_joints * 3            # 225
        
        assert self._num_face == 20, f"Expected 20 face landmarks, got {self._num_face}"
        assert self._feature_dim == 225, f"Expected 225 features, got {self._feature_dim}"
        
    @property
    def feature_dim(self) -> int:
        """Total feature dimension per frame."""
        return self._feature_dim
        
    def extract_from_video(self, video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
        """Extract keypoints from video file"""
        cap = cv2.VideoCapture(video_path)
        pose_data, left_hand_data, right_hand_data, face_data = [], [], [], []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
                
            # Flip for selfie view and convert to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            
            pose_data.append(self._extract_pose(results))
            left_hand_data.append(self._extract_left_hand(results))
            right_hand_data.append(self._extract_right_hand(results))
            face_data.append(self._extract_face(results))
                
        cap.release()
        
        return self._process_keypoints(pose_data, left_hand_data, right_hand_data, face_data, max_frames)
    
    def extract_from_frames(self, frame_dir: Path, max_frames: Optional[int] = None) -> np.ndarray:
        """Extract keypoints from directory of frames (PHOENIX format)"""
        frame_files = sorted(frame_dir.glob("*.png"), key=lambda x: int(x.stem.replace('images', '')))
        
        pose_data, left_hand_data, right_hand_data, face_data = [], [], [], []
        
        for frame_path in frame_files[:max_frames] if max_frames else frame_files:
            image = cv2.imread(str(frame_path))
            if image is None:
                print(f"Warning: Could not read {frame_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            
            pose_data.append(self._extract_pose(results))
            left_hand_data.append(self._extract_left_hand(results))
            right_hand_data.append(self._extract_right_hand(results))
            face_data.append(self._extract_face(results))
        
        if not pose_data:
            print(f"Warning: No frames found in {frame_dir}")
            return np.zeros((max_frames or 1, self._feature_dim))
        
        return self._process_keypoints(pose_data, left_hand_data, right_hand_data, face_data, max_frames)
    
    # ─── Per-frame landmark extraction helpers ───────────────────────
    
    def _extract_pose(self, results) -> List[Tuple[float, float, float]]:
        if results.pose_landmarks:
            return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return [(0.0, 0.0, 0.0)] * 33
    
    def _extract_left_hand(self, results) -> List[Tuple[float, float, float]]:
        if results.left_hand_landmarks:
            return [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]
        return [(0.0, 0.0, 0.0)] * 21
    
    def _extract_right_hand(self, results) -> List[Tuple[float, float, float]]:
        if results.right_hand_landmarks:
            return [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]
        return [(0.0, 0.0, 0.0)] * 21
    
    def _extract_face(self, results) -> List[Tuple[float, float, float]]:
        """Extract selected face landmarks from FaceMesh (468 total → 20 selected)."""
        if results.face_landmarks:
            all_lm = results.face_landmarks.landmark
            selected = []
            for idx in self.FACE_LANDMARKS_TO_KEEP:
                if idx < len(all_lm):
                    lm = all_lm[idx]
                    selected.append((lm.x, lm.y, lm.z))
                else:
                    selected.append((0.0, 0.0, 0.0))
            return selected
        return [(0.0, 0.0, 0.0)] * self._num_face
    
    # ─── Processing pipeline ─────────────────────────────────────────
    
    def _process_keypoints(self, pose_data, left_hand_data, right_hand_data, face_data, max_frames):
        """Apply interpolation, normalization, and feature engineering."""
        
        # Step 1: Interpolate missing frames (all four streams)
        pose_data = self._interpolate_keypoints(pose_data)
        left_hand_data = self._interpolate_keypoints(left_hand_data)
        right_hand_data = self._interpolate_keypoints(right_hand_data)
        face_data = self._interpolate_keypoints(face_data)
        
        # Step 2: Put hands in body coordinates (relative to wrists)
        pose_data, left_hand_data, right_hand_data = self._put_hands_in_body(
            pose_data, left_hand_data, right_hand_data
        )
        
        # Step 3: Convert pose to PoseNet format (keep only relevant landmarks)
        pose_data = self._mediapipe_to_posenet(pose_data)
        
        # Step 4: Apply chicken-neck normalization (center on neck)
        pose_data, left_hand_data, right_hand_data, face_data = self._chicken_neck_normalization(
            pose_data, left_hand_data, right_hand_data, face_data
        )
        
        # Convert to numpy
        pose_array = np.array(pose_data, dtype=np.float32)
        left_hand_array = np.array(left_hand_data, dtype=np.float32)
        right_hand_array = np.array(right_hand_data, dtype=np.float32)
        face_array = np.array(face_data, dtype=np.float32)
        
        # Ensure same frame count across all streams
        min_frames_count = min(
            pose_array.shape[0], left_hand_array.shape[0],
            right_hand_array.shape[0], face_array.shape[0]
        )
        pose_array = pose_array[:min_frames_count]
        left_hand_array = left_hand_array[:min_frames_count]
        right_hand_array = right_hand_array[:min_frames_count]
        face_array = face_array[:min_frames_count]
        
        # Flatten each to (T, joints*3)
        pose_flat = pose_array.reshape(min_frames_count, -1)       # (T, 39)
        left_flat = left_hand_array.reshape(min_frames_count, -1)  # (T, 63)
        right_flat = right_hand_array.reshape(min_frames_count, -1)# (T, 63)
        face_flat = face_array.reshape(min_frames_count, -1)       # (T, 60)
        
        # Concatenate: 39 + 63 + 63 + 60 = 225
        data = np.concatenate([pose_flat, left_flat, right_flat, face_flat], axis=1)
        
        assert data.shape[1] == self._feature_dim, (
            f"Expected {self._feature_dim} features, got {data.shape[1]}. "
            f"pose={pose_flat.shape[1]}, left={left_flat.shape[1]}, "
            f"right={right_flat.shape[1]}, face={face_flat.shape[1]}"
        )
        
        # Pad or truncate to max_frames
        if max_frames:
            if len(data) < max_frames:
                padding = np.zeros((max_frames - len(data), data.shape[1]))
                data = np.vstack([data, padding])
            else:
                data = data[:max_frames]
        
        # Replace NaN
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data)
        
        return data
    
    # ─── Interpolation ───────────────────────────────────────────────
    
    def _interpolate_keypoints(self, data: List) -> List:
        """Interpolate missing keypoints (all-zero frames) using neighbors."""
        if not data:
            return data
            
        data_array = np.array(data, dtype=np.float32)
        
        for i in range(len(data_array)):
            if np.all(data_array[i] == 0):
                last_valid = i - 1
                while last_valid >= 0 and np.all(data_array[last_valid] == 0):
                    last_valid -= 1
                    
                next_valid = i + 1
                while next_valid < len(data_array) and np.all(data_array[next_valid] == 0):
                    next_valid += 1
                    
                if last_valid >= 0 and next_valid < len(data_array):
                    alpha = (i - last_valid) / (next_valid - last_valid)
                    data_array[i] = (1 - alpha) * data_array[last_valid] + alpha * data_array[next_valid]
                elif last_valid >= 0:
                    data_array[i] = data_array[last_valid]
                elif next_valid < len(data_array):
                    data_array[i] = data_array[next_valid]
                    
        return data_array.tolist()
    
    # ─── Coordinate transforms ───────────────────────────────────────
    
    def _put_hands_in_body(self, pose, left_hand, right_hand):
        """Transform hand coordinates relative to wrists."""
        pose_array = np.array(pose, dtype=np.float32)
        left_hand_array = np.array(left_hand, dtype=np.float32)
        right_hand_array = np.array(right_hand, dtype=np.float32)
        
        if pose_array.shape[1] < 17:
            print(f"Warning: Pose has only {pose_array.shape[1]} landmarks, expected at least 17")
            return pose_array.tolist(), left_hand_array.tolist(), right_hand_array.tolist()
        
        for frame in range(len(pose_array)):
            # MediaPipe: left wrist = 15, right wrist = 16
            left_wrist = pose_array[frame, 15].copy()
            right_wrist = pose_array[frame, 16].copy()
            
            if frame < len(left_hand_array) and not np.all(left_hand_array[frame] == 0):
                left_hand_array[frame] = left_hand_array[frame] - left_wrist
                
            if frame < len(right_hand_array) and not np.all(right_hand_array[frame] == 0):
                right_hand_array[frame] = right_hand_array[frame] - right_wrist
                
        return pose_array.tolist(), left_hand_array.tolist(), right_hand_array.tolist()
    
    def _mediapipe_to_posenet(self, pose):
        """Keep only the 13 selected pose landmarks."""
        pose_array = np.array(pose, dtype=np.float32)
        
        max_idx = max(self.POSE_LANDMARKS_TO_KEEP)
        if pose_array.shape[1] < max_idx + 1:
            padding = np.zeros((pose_array.shape[0], max_idx + 1 - pose_array.shape[1], 3))
            pose_array = np.concatenate([pose_array, padding], axis=1)
        
        selected = pose_array[:, self.POSE_LANDMARKS_TO_KEEP]
        return selected.tolist()
    
    def _chicken_neck_normalization(self, pose, left_hand, right_hand, face):
        """
        Normalize all landmarks around the neck point (midpoint of shoulders).
        After _mediapipe_to_posenet, the selected pose indices are:
            [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
        mapped to positions 0..12.
        Index 5 in POSE_LANDMARKS_TO_KEEP = original landmark 11 (left shoulder)
        Index 6 in POSE_LANDMARKS_TO_KEEP = original landmark 12 (right shoulder)
        So neck = average of selected[5] and selected[6].
        """
        pose_array = np.array(pose, dtype=np.float32)
        left_hand_array = np.array(left_hand, dtype=np.float32)
        right_hand_array = np.array(right_hand, dtype=np.float32)
        face_array = np.array(face, dtype=np.float32)
        
        # Shoulder indices in the *selected* 13-landmark array
        LEFT_SHOULDER_SEL = 5   # original landmark 11
        RIGHT_SHOULDER_SEL = 6  # original landmark 12
        
        for frame in range(len(pose_array)):
            if pose_array.shape[1] > RIGHT_SHOULDER_SEL:
                neck = (pose_array[frame, LEFT_SHOULDER_SEL] + 
                        pose_array[frame, RIGHT_SHOULDER_SEL]) * 0.5
                
                pose_array[frame] -= neck
                
                if frame < len(left_hand_array) and not np.all(left_hand_array[frame] == 0):
                    left_hand_array[frame] -= neck
                    
                if frame < len(right_hand_array) and not np.all(right_hand_array[frame] == 0):
                    right_hand_array[frame] -= neck
                
                # Also normalize face landmarks relative to neck
                if frame < len(face_array) and not np.all(face_array[frame] == 0):
                    face_array[frame] -= neck
                    
        return pose_array.tolist(), left_hand_array.tolist(), right_hand_array.tolist(), face_array.tolist()
    
    def close(self):
        self.holistic.close()
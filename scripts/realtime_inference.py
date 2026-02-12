"""
Real-time inference pipeline for MultiStream SLT.

Processes live video (webcam or file) using MediaPipe Holistic for
keypoint extraction and hand detection, then runs the MultiStreamSLT model.

Usage:
    python -m scripts.realtime_inference --checkpoint model.pt --source 0  # webcam
    python -m scripts.realtime_inference --checkpoint model.pt --source video.mp4
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

from torchvision import transforms


class RealtimeMediaPipeExtractor:
    """
    Optimized MediaPipe extraction for real-time inference.
    Extracts keypoints + hand bounding boxes in a single Holistic pass.
    (Avoids running separate Hands + Holistic models like the preprocessing scripts.)
    """

    FACE_INDICES = sorted(set(
        list(range(61, 68)) + [291, 308, 324, 318, 402, 317, 14, 87, 178, 88] +
        list(range(78, 82)) + [13, 312, 311, 310, 415] +
        list(range(133, 145)) + [33, 160, 158, 133, 153, 144] +
        list(range(362, 374)) + [263, 387, 385, 362, 380, 373] +
        list(range(70, 80)) + [46, 52, 53, 63, 66, 65] +
        list(range(300, 310)) + [276, 282, 283, 293, 296, 295] +
        [1, 2, 98, 327] + list(range(195, 199))
    ))[:80]

    def __init__(self, hand_crop_size: Tuple[int, int] = (112, 112), hand_padding: float = 1.5):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_crop_size = hand_crop_size
        self.hand_padding = hand_padding

    def process_frame(self, frame_rgb: np.ndarray) -> Dict:
        """
        Single-pass extraction of keypoints + hand crops.
        Returns dict with 'kpts' (498,), 'left_hand' (H,W,3), 'right_hand' (H,W,3).
        """
        h, w = frame_rgb.shape[:2]
        results = self.holistic.process(frame_rgb)

        # ── Keypoints (same logic as preprocess_kpts.py) ──
        # Pose: 33*4
        if results.pose_landmarks:
            pose = []
            for lm in results.pose_landmarks.landmark:
                pose.extend([lm.x, lm.y, lm.z, lm.visibility])
            pose = np.array(pose, dtype=np.float32)
        else:
            pose = np.zeros(132, dtype=np.float32)

        # Left hand: 21*3
        if results.left_hand_landmarks:
            lh = []
            for lm in results.left_hand_landmarks.landmark:
                lh.extend([lm.x, lm.y, lm.z])
            lh = np.array(lh, dtype=np.float32)
        else:
            lh = np.zeros(63, dtype=np.float32)

        # Right hand: 21*3
        if results.right_hand_landmarks:
            rh = []
            for lm in results.right_hand_landmarks.landmark:
                rh.extend([lm.x, lm.y, lm.z])
            rh = np.array(rh, dtype=np.float32)
        else:
            rh = np.zeros(63, dtype=np.float32)

        # Face: 80*3
        if results.face_landmarks:
            face = []
            for idx in self.FACE_INDICES:
                lm = results.face_landmarks.landmark[idx]
                face.extend([lm.x, lm.y, lm.z])
            face = np.array(face, dtype=np.float32)
        else:
            face = np.zeros(240, dtype=np.float32)

        kpts = np.concatenate([pose, lh, rh, face])  # (498,)

        # ── Hand crops (same logic as preprocess_hands.py) ──
        left_crop = self._extract_hand_crop(frame_rgb, results.left_hand_landmarks, h, w)
        right_crop = self._extract_hand_crop(frame_rgb, results.right_hand_landmarks, h, w)

        return {
            "kpts": kpts,
            "left_hand": left_crop,
            "right_hand": right_crop,
        }

    def _extract_hand_crop(
        self, frame: np.ndarray, landmarks, h: int, w: int
    ) -> np.ndarray:
        """Crop and resize hand region from frame."""
        H, W = self.hand_crop_size
        if landmarks is None:
            return np.zeros((H, W, 3), dtype=np.uint8)

        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        pad = self.hand_padding
        bw = max(1.0, (x_max - x_min) * pad)
        bh = max(1.0, (y_max - y_min) * pad)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    def close(self):
        self.holistic.close()


class RealtimeSLTPipeline:
    """
    Buffered real-time inference pipeline.
    Accumulates frames in a sliding window, runs model when buffer is full.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bart_tokenizer,
        gloss_tokenizer=None,
        device: str = "cuda",
        num_frames: int = 64,
        rgb_size: Tuple[int, int] = (224, 224),
        hand_crop_size: Tuple[int, int] = (112, 112),
    ):
        self.model = model.to(device).eval()
        self.bart_tokenizer = bart_tokenizer
        self.gloss_tokenizer = gloss_tokenizer
        self.device = device
        self.num_frames = num_frames

        self.extractor = RealtimeMediaPipeExtractor(hand_crop_size)

        # RGB transform
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(rgb_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Frame buffers
        self.rgb_buffer: deque = deque(maxlen=num_frames)
        self.hands_buffer: deque = deque(maxlen=num_frames)
        self.kpts_buffer: deque = deque(maxlen=num_frames)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        Process a single video frame.
        Returns translation result when buffer is full, else None.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Extract features
        extracted = self.extractor.process_frame(frame_rgb)

        # RGB: transform to tensor
        rgb_tensor = self.rgb_transform(frame_rgb)  # (3, H, W)
        self.rgb_buffer.append(rgb_tensor)

        # Hands: stack left + right, normalize
        left = torch.from_numpy(extracted["left_hand"]).permute(2, 0, 1).float() / 255.0
        right = torch.from_numpy(extracted["right_hand"]).permute(2, 0, 1).float() / 255.0
        hands_tensor = torch.stack([left, right], dim=0)  # (2, 3, H, W)
        self.hands_buffer.append(hands_tensor)

        # Keypoints: normalize
        kpts = torch.from_numpy(extracted["kpts"]).float()
        mean = kpts.mean()
        std = kpts.std() + 1e-6
        kpts = (kpts - mean) / std
        self.kpts_buffer.append(kpts)

        # Run inference when buffer is full
        if len(self.rgb_buffer) >= self.num_frames:
            return self._run_inference()
        return None

    @torch.no_grad()
    def _run_inference(self) -> Dict:
        """Run model on current buffer contents."""
        # Stack buffers → (1, T, ...)
        rgb = torch.stack(list(self.rgb_buffer), dim=0).unsqueeze(0).to(self.device)
        hands = torch.stack(list(self.hands_buffer), dim=0).unsqueeze(0).to(self.device)
        kpts = torch.stack(list(self.kpts_buffer), dim=0).unsqueeze(0).to(self.device)

        # Translate
        token_ids = self.model.translate(rgb, hands, kpts, beam_width=5)
        translation = self.bart_tokenizer.decode(token_ids[0], skip_special_tokens=True)

        # Gloss recognition
        gloss_ids = self.model.recognize_glosses(rgb, hands, kpts)
        glosses = ""
        if self.gloss_tokenizer is not None:
            # Simple CTC decode on the raw predictions
            from utils.tokenizer import ctc_greedy_decode
            gloss_logits = self.model.ctc_head(self.model.encode(rgb, hands, kpts))
            decoded = ctc_greedy_decode(gloss_logits)
            glosses = self.gloss_tokenizer.decode(decoded[0])

        # Clear half the buffer for sliding window
        for _ in range(self.num_frames // 2):
            if self.rgb_buffer:
                self.rgb_buffer.popleft()
                self.hands_buffer.popleft()
                self.kpts_buffer.popleft()

        return {
            "translation": translation,
            "glosses": glosses,
        }

    def close(self):
        self.extractor.close()


def run_realtime(
    checkpoint_path: str,
    source=0,
    device: str = "cuda",
):
    """
    Run real-time SLT from webcam or video file.

    Args:
        checkpoint_path: Path to model checkpoint
        source: 0 for webcam, or path to video file
        device: 'cuda' or 'cpu'
    """
    from transformers import BartTokenizer
    from models import MultiStreamSLT
    from utils import GlossTokenizer

    # Load model
    print("Loading model...")
    model = MultiStreamSLT()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    gloss_tokenizer = GlossTokenizer()
    if "gloss_tokenizer" in state:
        gloss_tokenizer.word2idx = state["gloss_tokenizer"]["word2idx"]
        gloss_tokenizer.idx2word = {int(v): k for k, v in gloss_tokenizer.word2idx.items()}

    pipeline = RealtimeSLTPipeline(
        model=model,
        bart_tokenizer=bart_tokenizer,
        gloss_tokenizer=gloss_tokenizer,
        device=device,
    )

    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process_frame(frame)

            # Display
            display = frame.copy()
            if result:
                cv2.putText(display, f"Gloss: {result['glosses']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Trans: {result['translation']}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                buf_len = len(pipeline.rgb_buffer)
                cv2.putText(display, f"Buffering... {buf_len}/{pipeline.num_frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            cv2.imshow("MultiStream SLT", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_realtime(args.checkpoint, source, args.device)
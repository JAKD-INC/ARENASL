"""Extract unified-schema landmark frames from a video using MediaPipe Tasks."""
import os
# Silence MediaPipe/TFLite's harmless per-init chatter ("Feedback manager ...",
# XNNPACK delegate INFO). Must be set before importing mediapipe. 2 = errors only.
os.environ.setdefault("GLOG_minloglevel", "2")
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision
from asl.schema import assemble_frame

_MODELS = Path(__file__).resolve().parents[2] / "public" / "models"


def _landmarkers():
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpy.BaseOptions(model_asset_path=str(_MODELS / "hand_landmarker.task")),
        num_hands=2, running_mode=vision.RunningMode.VIDEO))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpy.BaseOptions(model_asset_path=str(_MODELS / "pose_landmarker_lite.task")),
        num_poses=1, running_mode=vision.RunningMode.VIDEO))
    return hand, pose


def _xyz(landmarks):
    return [[lm.x, lm.y, lm.z] for lm in landmarks]


def extract_frames(video_path) -> list[np.ndarray]:
    """Return a list of (49,3) assembled frames; frames with no pose are skipped."""
    hand, pose = _landmarkers()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i / fps * 1000)
            hr = hand.detect_for_video(image, ts)
            pr = pose.detect_for_video(image, ts)
            i += 1
            if not pr.pose_landmarks:
                continue
            left = right = None
            for lms, handed in zip(hr.hand_landmarks, hr.handedness):
                if handed[0].category_name == "Left":
                    left = _xyz(lms)
                else:
                    right = _xyz(lms)
            frames.append(assemble_frame(_xyz(pr.pose_landmarks[0]), left, right))
    finally:
        cap.release()
        hand.close()
        pose.close()
    return frames

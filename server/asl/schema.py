import numpy as np

# MediaPipe Pose landmark indices we keep, in schema slot order.
# Slots 0 and 1 MUST be the shoulders (see asl.features.LEFT/RIGHT_SHOULDER).
POSE_INDICES = [11, 12, 13, 14, 15, 16, 0]  # L/R shoulder, L/R elbow, L/R wrist, nose
N_POSE = len(POSE_INDICES)
N_HAND = 21
N_KEYPOINTS = N_POSE + 2 * N_HAND  # 7 + 42 = 49


def assemble_frame(pose, hand_left, hand_right) -> np.ndarray:
    """Assemble MediaPipe landmark lists into the unified (49, 3) schema.

    Args:
        pose: list of 33 [x, y, z] pose landmarks, or None if no person detected.
        hand_left / hand_right: list of 21 [x, y, z] hand landmarks, or None if
            that hand was not detected (slots are zero-filled).

    Returns:
        (N_KEYPOINTS, 3) float array.

    Raises:
        ValueError: if pose is None (no usable frame), or if the landmark counts
            are wrong (guards against silently corrupted upstream input).
    """
    if pose is None:
        raise ValueError("no pose landmarks; frame unusable")
    if len(pose) != 33:
        raise ValueError(f"pose must have 33 landmarks, got {len(pose)}")
    for label, hand in (("hand_left", hand_left), ("hand_right", hand_right)):
        if hand is not None and len(hand) != N_HAND:
            raise ValueError(f"{label} must have {N_HAND} landmarks, got {len(hand)}")
    frame = np.zeros((N_KEYPOINTS, 3), dtype=float)
    for slot, pose_idx in enumerate(POSE_INDICES):
        frame[slot] = pose[pose_idx]
    if hand_left is not None:
        frame[N_POSE:N_POSE + N_HAND] = hand_left
    if hand_right is not None:
        frame[N_POSE + N_HAND:N_KEYPOINTS] = hand_right
    return frame

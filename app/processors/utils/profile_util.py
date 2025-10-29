"""
Profile Face Utility Functions

This module provides specialized functions for handling profile (side-view) face swapping.
Profile faces present unique challenges compared to frontal faces due to:
- Different keypoint distributions (only one eye visible)
- Asymmetric face structure
- Different depth cues and occlusion patterns
"""

import numpy as np
import torch
from torchvision.transforms import v2


def detect_profile_orientation(kps_5: np.ndarray, threshold: float = 0.3) -> str:
    """
    Detect if a face is in profile and which direction it's facing.

    Args:
        kps_5: 5-point face landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        threshold: Sensitivity threshold for profile detection (0.0-1.0)

    Returns:
        'frontal', 'left_profile', or 'right_profile'
    """
    # Calculate the horizontal distance between eyes
    left_eye = kps_5[0]
    right_eye = kps_5[1]
    nose = kps_5[2]

    eye_distance = np.abs(right_eye[0] - left_eye[0])

    # Calculate nose position relative to eyes
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset = nose[0] - eye_center_x

    # If eyes are very close together, it's likely a profile
    if eye_distance < 20:  # Threshold for very close eyes
        # Determine direction based on which eye is more visible
        # and nose position
        if nose_offset > 0:
            return 'right_profile'
        else:
            return 'left_profile'

    # Calculate the ratio of nose offset to eye distance
    offset_ratio = abs(nose_offset) / (eye_distance + 1e-6)

    if offset_ratio > threshold:
        if nose_offset > 0:
            return 'right_profile'
        else:
            return 'left_profile'

    return 'frontal'


def adjust_profile_keypoints(kps_5: np.ndarray, profile_side: str, enhancement: float = 0.5) -> np.ndarray:
    """
    Adjust keypoints specifically for profile faces to improve alignment.

    Args:
        kps_5: 5-point face landmarks
        profile_side: 'left_profile' or 'right_profile'
        enhancement: Enhancement strength (0.0-1.0)

    Returns:
        Adjusted keypoints
    """
    kps_adjusted = kps_5.copy()

    if profile_side == 'left_profile':
        # Left profile: left side of face is visible
        # The right eye is likely occluded or barely visible
        # Adjust right eye position based on left eye
        left_eye = kps_adjusted[0]
        right_eye = kps_adjusted[1]
        nose = kps_adjusted[2]

        # Estimate better position for occluded right eye
        eye_y_diff = right_eye[1] - left_eye[1]
        eye_x_diff = right_eye[0] - left_eye[0]

        # For left profile, push right eye further right to improve alignment
        adjustment = enhancement * 10
        kps_adjusted[1][0] = right_eye[0] + adjustment

        # Adjust mouth points similarly
        kps_adjusted[4][0] = kps_adjusted[4][0] + adjustment * 0.5

    elif profile_side == 'right_profile':
        # Right profile: right side of face is visible
        # The left eye is likely occluded or barely visible
        left_eye = kps_adjusted[0]
        right_eye = kps_adjusted[1]

        # For right profile, push left eye further left to improve alignment
        adjustment = enhancement * 10
        kps_adjusted[0][0] = left_eye[0] - adjustment

        # Adjust mouth points similarly
        kps_adjusted[3][0] = kps_adjusted[3][0] - adjustment * 0.5

    return kps_adjusted


def create_profile_mask(profile_side: str, enhancement: float = 0.5, device='cuda') -> torch.Tensor:
    """
    Create an asymmetric mask for profile face blending.

    Args:
        profile_side: 'left_profile' or 'right_profile'
        enhancement: Enhancement strength (0.0-1.0)
        device: torch device

    Returns:
        Profile mask tensor (128x128)
    """
    # Create base circular mask
    y, x = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing='ij')
    center_x, center_y = 64, 64

    # Calculate distance from center
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create elliptical mask that favors the visible side
    if profile_side == 'left_profile':
        # For left profile, the left side is more visible
        # Create an asymmetric mask that preserves more of the left side
        x_weight = torch.where(x < center_x,
                               1.0 + enhancement * 0.3,  # Boost left side
                               1.0 - enhancement * 0.2)   # Reduce right side
    elif profile_side == 'right_profile':
        # For right profile, the right side is more visible
        x_weight = torch.where(x > center_x,
                               1.0 + enhancement * 0.3,  # Boost right side
                               1.0 - enhancement * 0.2)   # Reduce left side
    else:
        x_weight = torch.ones_like(x)

    # Apply weighted distance
    weighted_dist = dist / x_weight

    # Create smooth falloff
    mask = torch.clamp((64 - weighted_dist) / 10, 0, 1)

    return mask.unsqueeze(0)  # Add channel dimension


def apply_profile_color_correction(swap: torch.Tensor, original: torch.Tensor,
                                   profile_side: str, enhancement: float = 0.5) -> torch.Tensor:
    """
    Apply profile-specific color correction that accounts for lighting differences
    between the visible and occluded sides of the face.

    Args:
        swap: Swapped face tensor (3, 512, 512)
        original: Original face tensor (3, 512, 512)
        profile_side: 'left_profile' or 'right_profile'
        enhancement: Enhancement strength (0.0-1.0)

    Returns:
        Color-corrected swap tensor
    """
    # Create side-specific weight masks
    if profile_side == 'left_profile' or profile_side == 'right_profile':
        # Calculate mean colors on visible side
        mid = swap.shape[2] // 2

        if profile_side == 'left_profile':
            # Focus on left side for color matching
            visible_region_orig = original[:, :, :mid]
            visible_region_swap = swap[:, :, :mid]
        else:
            # Focus on right side for color matching
            visible_region_orig = original[:, :, mid:]
            visible_region_swap = swap[:, :, mid:]

        # Calculate color statistics for visible region
        orig_mean = visible_region_orig.float().mean(dim=[1, 2], keepdim=True)
        swap_mean = visible_region_swap.float().mean(dim=[1, 2], keepdim=True)
        orig_std = visible_region_orig.float().std(dim=[1, 2], keepdim=True)
        swap_std = visible_region_swap.float().std(dim=[1, 2], keepdim=True)

        # Apply color transfer
        swap_normalized = (swap.float() - swap_mean) / (swap_std + 1e-6)
        swap_corrected = swap_normalized * orig_std + orig_mean

        # Blend with original based on enhancement
        swap_blended = swap * (1 - enhancement * 0.3) + swap_corrected * (enhancement * 0.3)
        swap = torch.clamp(swap_blended, 0, 255)

    return swap


def enhance_profile_landmarks(kps_5: np.ndarray, kps_all: np.ndarray,
                              profile_side: str, enhancement: float = 0.5):
    """
    Enhance full landmark set for profile faces.

    Args:
        kps_5: 5-point landmarks
        kps_all: Full landmark set (68/203/478 points)
        profile_side: Profile orientation
        enhancement: Enhancement strength

    Returns:
        Enhanced full landmarks
    """
    if kps_all is None or len(kps_all) == 0:
        return kps_all

    kps_enhanced = kps_all.copy()

    # For profile faces, landmarks on the occluded side may be less accurate
    # Apply smoothing or adjustment based on visible side landmarks
    if profile_side == 'left_profile':
        # For left profile, landmarks on the right side need adjustment
        # This is a simplified approach - in practice, you might use more
        # sophisticated landmark correction based on the specific landmark model
        pass
    elif profile_side == 'right_profile':
        # For right profile, landmarks on the left side need adjustment
        pass

    return kps_enhanced


def get_profile_border_adjustments(profile_side: str, enhancement: float = 0.5) -> dict:
    """
    Get border mask adjustments for profile faces.

    Args:
        profile_side: Profile orientation
        enhancement: Enhancement strength

    Returns:
        Dictionary with adjusted border values
    """
    # Base border values
    adjustments = {
        'BorderTopSlider': 10,
        'BorderBottomSlider': 10,
        'BorderLeftSlider': 10,
        'BorderRightSlider': 10,
    }

    # Adjust borders based on profile side
    # We want to extend the border on the occluded side to better blend
    if profile_side == 'left_profile':
        # Extend right border for left profile
        adjustments['BorderRightSlider'] = int(10 + enhancement * 15)
        adjustments['BorderLeftSlider'] = int(10 - enhancement * 5)
    elif profile_side == 'right_profile':
        # Extend left border for right profile
        adjustments['BorderLeftSlider'] = int(10 + enhancement * 15)
        adjustments['BorderRightSlider'] = int(10 - enhancement * 5)

    return adjustments

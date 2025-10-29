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


def detect_profile_orientation(kps_5: np.ndarray, threshold: float = 0.3) -> tuple:
    """
    Detect face orientation and estimate angle from frontal view.

    Args:
        kps_5: 5-point face landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        threshold: Sensitivity threshold for profile detection (0.0-1.0)

    Returns:
        tuple: (orientation, angle_degrees, confidence)
        orientation: 'frontal', 'three_quarter_left', 'three_quarter_right', 'left_profile', 'right_profile'
        angle_degrees: Estimated angle from frontal view (0-90 degrees)
        confidence: Detection confidence (0.0-1.0)
    """
    # Calculate the horizontal distance between eyes
    left_eye = kps_5[0]
    right_eye = kps_5[1]
    nose = kps_5[2]

    eye_distance = np.abs(right_eye[0] - left_eye[0])

    # Calculate nose position relative to eyes
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset = nose[0] - eye_center_x

    # Calculate the ratio of nose offset to eye distance
    offset_ratio = abs(nose_offset) / (eye_distance + 1e-6)

    # Estimate angle based on offset ratio and eye distance
    # For a pure frontal face: offset_ratio ≈ 0, eye_distance is large
    # For a pure profile: offset_ratio is large, eye_distance is small

    # Normalize eye distance to expected frontal range (typically 60-100 pixels)
    eye_distance_norm = np.clip(eye_distance / 80.0, 0.2, 1.0)

    # Calculate angle estimation
    # offset_ratio of 0.0 = 0°, offset_ratio of 1.5+ = 90°
    angle_from_offset = np.clip(offset_ratio * 60, 0, 90)

    # Eye distance compression also indicates rotation
    # eye_distance_norm of 1.0 = 0° contribution, 0.2 = +30° contribution
    angle_from_eyes = (1.0 - eye_distance_norm) * 40

    # Combined angle estimate (weighted average)
    angle = np.clip(angle_from_offset * 0.7 + angle_from_eyes * 0.3, 0, 90)

    # Calculate confidence based on consistency between metrics
    angle_diff = abs(angle_from_offset - angle_from_eyes)
    confidence = np.clip(1.0 - (angle_diff / 60.0), 0.5, 1.0)

    # Determine orientation category based on angle
    direction = 'right' if nose_offset > 0 else 'left'

    if angle < 15:
        orientation = 'frontal'
    elif angle < 45:
        orientation = f'three_quarter_{direction}'
    else:
        orientation = f'{direction}_profile'

    return orientation, float(angle), float(confidence)


def calculate_adaptive_enhancement(angle: float, user_enhancement: float = 0.5,
                                   orientation: str = 'frontal') -> float:
    """
    Calculate adaptive enhancement multiplier based on detected angle.

    This function automatically adjusts the enhancement strength based on how much
    the face is rotated from frontal view. More extreme angles need stronger adjustments.

    Args:
        angle: Detected angle in degrees (0-90)
        user_enhancement: User-specified enhancement value (0.0-1.0)
        orientation: Face orientation classification

    Returns:
        Adjusted enhancement value (0.0-1.0)
    """
    # Define angle-based multipliers
    if angle < 15:
        # Frontal faces: minimal enhancement needed
        angle_multiplier = 0.3
    elif angle < 30:
        # Slight angle: subtle enhancement
        angle_multiplier = 0.5
    elif angle < 45:
        # Three-quarter view: moderate enhancement
        angle_multiplier = 0.75
    elif angle < 60:
        # Strong three-quarter: significant enhancement
        angle_multiplier = 0.9
    else:
        # Profile (60-90°): maximum enhancement
        angle_multiplier = 1.0

    # Combine user preference with adaptive multiplier
    # User can still control overall strength, but it's scaled appropriately
    adaptive_enhancement = user_enhancement * angle_multiplier

    return float(np.clip(adaptive_enhancement, 0.0, 1.0))


def adjust_profile_keypoints(kps_5: np.ndarray, profile_side: str, enhancement: float = 0.5,
                            angle: float = 0.0) -> np.ndarray:
    """
    Adjust keypoints specifically for profile and three-quarter faces to improve alignment.

    Args:
        kps_5: 5-point face landmarks
        profile_side: 'left_profile', 'right_profile', 'three_quarter_left', 'three_quarter_right', or 'frontal'
        enhancement: Enhancement strength (0.0-1.0)
        angle: Face angle in degrees (0-90), used to scale adjustments

    Returns:
        Adjusted keypoints
    """
    kps_adjusted = kps_5.copy()

    # Scale adjustment based on angle (more angle = more adjustment needed)
    angle_scale = np.clip(angle / 60.0, 0.0, 1.0)  # 60° = full scale

    if profile_side in ['left_profile', 'three_quarter_left']:
        # Left side visible: adjust right (occluded) side
        left_eye = kps_adjusted[0]
        right_eye = kps_adjusted[1]
        nose = kps_adjusted[2]

        # For left profile/three-quarter, push right eye further right
        # Three-quarter gets scaled down adjustment (less extreme than profile)
        adjustment = enhancement * 10 * angle_scale
        kps_adjusted[1][0] = right_eye[0] + adjustment

        # Adjust mouth points similarly
        kps_adjusted[4][0] = kps_adjusted[4][0] + adjustment * 0.5

    elif profile_side in ['right_profile', 'three_quarter_right']:
        # Right side visible: adjust left (occluded) side
        left_eye = kps_adjusted[0]
        right_eye = kps_adjusted[1]

        # For right profile/three-quarter, push left eye further left
        adjustment = enhancement * 10 * angle_scale
        kps_adjusted[0][0] = left_eye[0] - adjustment

        # Adjust mouth points similarly
        kps_adjusted[3][0] = kps_adjusted[3][0] - adjustment * 0.5

    # No adjustment for frontal faces
    return kps_adjusted


def create_profile_mask(profile_side: str, enhancement: float = 0.5, device='cuda',
                       angle: float = 0.0) -> torch.Tensor:
    """
    Create an asymmetric mask for profile and three-quarter face blending.

    Args:
        profile_side: 'left_profile', 'right_profile', 'three_quarter_left', 'three_quarter_right', or 'frontal'
        enhancement: Enhancement strength (0.0-1.0)
        device: torch device
        angle: Face angle in degrees (0-90), used to scale asymmetry

    Returns:
        Profile mask tensor (128x128)
    """
    # Create base circular mask
    y, x = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing='ij')
    center_x, center_y = 64, 64

    # Calculate distance from center
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Scale asymmetry based on angle (more angle = more asymmetry)
    angle_scale = np.clip(angle / 60.0, 0.0, 1.0)  # 60° = full asymmetry

    # Create elliptical mask that favors the visible side
    if profile_side in ['left_profile', 'three_quarter_left']:
        # For left side visible, preserve more of the left side
        boost_factor = enhancement * 0.3 * angle_scale
        reduce_factor = enhancement * 0.2 * angle_scale
        x_weight = torch.where(x < center_x,
                               1.0 + boost_factor,   # Boost left side
                               1.0 - reduce_factor)   # Reduce right side
    elif profile_side in ['right_profile', 'three_quarter_right']:
        # For right side visible, preserve more of the right side
        boost_factor = enhancement * 0.3 * angle_scale
        reduce_factor = enhancement * 0.2 * angle_scale
        x_weight = torch.where(x > center_x,
                               1.0 + boost_factor,   # Boost right side
                               1.0 - reduce_factor)   # Reduce left side
    else:
        # Frontal or unknown: symmetric mask
        x_weight = torch.ones_like(x)

    # Apply weighted distance
    weighted_dist = dist / x_weight

    # Create smooth falloff
    mask = torch.clamp((64 - weighted_dist) / 10, 0, 1)

    return mask.unsqueeze(0)  # Add channel dimension


def apply_profile_color_correction(swap: torch.Tensor, original: torch.Tensor,
                                   profile_side: str, enhancement: float = 0.5,
                                   angle: float = 0.0) -> torch.Tensor:
    """
    Apply profile-specific color correction that accounts for lighting differences
    between the visible and occluded sides of the face.

    Args:
        swap: Swapped face tensor (3, 512, 512)
        original: Original face tensor (3, 512, 512)
        profile_side: 'left_profile', 'right_profile', 'three_quarter_left', 'three_quarter_right', or 'frontal'
        enhancement: Enhancement strength (0.0-1.0)
        angle: Face angle in degrees (0-90), used to scale correction strength

    Returns:
        Color-corrected swap tensor
    """
    # Only apply for non-frontal faces
    if profile_side in ['left_profile', 'right_profile', 'three_quarter_left', 'three_quarter_right']:
        # Calculate mean colors on visible side
        mid = swap.shape[2] // 2

        if profile_side in ['left_profile', 'three_quarter_left']:
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

        # Scale correction strength based on angle
        angle_scale = np.clip(angle / 60.0, 0.0, 1.0)
        correction_strength = enhancement * 0.3 * angle_scale

        # Blend with original based on enhancement
        swap_blended = swap * (1 - correction_strength) + swap_corrected * correction_strength
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


def get_profile_border_adjustments(profile_side: str, enhancement: float = 0.5,
                                  angle: float = 0.0) -> dict:
    """
    Get border mask adjustments for profile and three-quarter faces.

    Args:
        profile_side: Profile orientation
        enhancement: Enhancement strength
        angle: Face angle in degrees (0-90), used to scale border adjustments

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

    # Scale border adjustments based on angle
    angle_scale = np.clip(angle / 60.0, 0.0, 1.0)

    # Adjust borders based on profile side
    # We want to extend the border on the occluded side to better blend
    if profile_side in ['left_profile', 'three_quarter_left']:
        # Extend right border for left-facing faces
        adjustments['BorderRightSlider'] = int(10 + enhancement * 15 * angle_scale)
        adjustments['BorderLeftSlider'] = max(5, int(10 - enhancement * 5 * angle_scale))
    elif profile_side in ['right_profile', 'three_quarter_right']:
        # Extend left border for right-facing faces
        adjustments['BorderLeftSlider'] = int(10 + enhancement * 15 * angle_scale)
        adjustments['BorderRightSlider'] = max(5, int(10 - enhancement * 5 * angle_scale))

    return adjustments

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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class WarningLevel(Enum):
    """Warning severity levels for UI display"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProfileWarning:
    """Data class for profile-related warnings"""
    level: WarningLevel
    title: str
    message: str
    recommendation: str
    dismissible: bool = True
    show_icon: bool = True
    action_button: Optional[str] = None
    action_callback: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'recommendation': self.recommendation,
            'dismissible': self.dismissible,
            'show_icon': self.show_icon,
            'action_button': self.action_button,
            'action_callback': self.action_callback,
        }


def generate_profile_warnings(source_orientation: str, source_angle: float,
                              target_orientation: str, target_angle: float,
                              source_confidence: float = 1.0,
                              target_confidence: float = 1.0) -> List[ProfileWarning]:
    """
    Generate real-time UI warnings for profile mismatches.

    Args:
        source_orientation: Source face orientation
        source_angle: Source face angle
        target_orientation: Target face orientation
        target_angle: Target face angle
        source_confidence: Source detection confidence
        target_confidence: Target detection confidence

    Returns:
        List of ProfileWarning objects for UI display
    """
    warnings = []

    # Check mismatch
    mismatch_info = check_profile_mismatch(
        source_orientation, target_orientation,
        source_angle, target_angle
    )

    if mismatch_info['has_mismatch']:
        severity = mismatch_info['severity']

        if severity == 'high':
            warning = ProfileWarning(
                level=WarningLevel.HIGH,
                title="⚠️ Major Orientation Mismatch",
                message=mismatch_info['message'],
                recommendation=mismatch_info['recommendation'],
                dismissible=False,
                action_button="Auto-Fix" if 'direction' in mismatch_info['message'].lower() else None,
                action_callback="flip_source_image" if 'direction' in mismatch_info['message'].lower() else None
            )
            warnings.append(warning)
        elif severity == 'medium':
            warning = ProfileWarning(
                level=WarningLevel.MEDIUM,
                title="⚡ Orientation Mismatch",
                message=mismatch_info['message'],
                recommendation=mismatch_info['recommendation'],
                dismissible=True,
                action_button="Auto-Flip" if 'direction' in mismatch_info['message'].lower() else None,
                action_callback="flip_source_image" if 'direction' in mismatch_info['message'].lower() else None
            )
            warnings.append(warning)
        elif severity == 'low':
            warning = ProfileWarning(
                level=WarningLevel.LOW,
                title="ℹ️ Minor Mismatch",
                message=mismatch_info['message'],
                recommendation=mismatch_info['recommendation'],
                dismissible=True
            )
            warnings.append(warning)

    # Check low confidence
    if source_confidence < 0.6:
        warning = ProfileWarning(
            level=WarningLevel.MEDIUM,
            title="⚠️ Low Source Detection Confidence",
            message=f"Source face detected with {source_confidence*100:.0f}% confidence ({source_orientation}, {source_angle:.0f}°)",
            recommendation="Consider using a clearer source image with better face visibility",
            dismissible=True
        )
        warnings.append(warning)

    if target_confidence < 0.6:
        warning = ProfileWarning(
            level=WarningLevel.MEDIUM,
            title="⚠️ Low Target Detection Confidence",
            message=f"Target face detected with {target_confidence*100:.0f}% confidence ({target_orientation}, {target_angle:.0f}°)",
            recommendation="Face detection may be uncertain. Consider adjusting detection threshold.",
            dismissible=True
        )
        warnings.append(warning)

    # Extreme angles warning
    if target_angle > 75 or source_angle > 75:
        warning = ProfileWarning(
            level=WarningLevel.INFO,
            title="💡 Extreme Profile Angle",
            message=f"Very extreme profile detected (Source: {source_angle:.0f}°, Target: {target_angle:.0f}°)",
            recommendation="Use 'Extreme Profile' preset for best results with angles > 70°",
            dismissible=True,
            action_button="Apply Preset",
            action_callback="apply_extreme_profile_preset"
        )
        warnings.append(warning)

    # Three-quarter optimization suggestion
    if 30 <= target_angle <= 50 and 'three_quarter' not in target_orientation:
        warning = ProfileWarning(
            level=WarningLevel.INFO,
            title="💡 Three-Quarter View Detected",
            message=f"Face at {target_angle:.0f}° - ideal for Three-Quarter Enhanced preset",
            recommendation="Consider using 'Three-Quarter Enhanced' preset for optimal results",
            dismissible=True,
            action_button="Apply Preset",
            action_callback="apply_three_quarter_preset"
        )
        warnings.append(warning)

    return warnings


def format_warning_for_display(warning: ProfileWarning, format_type: str = 'html') -> str:
    """
    Format warning for different display types.

    Args:
        warning: ProfileWarning object
        format_type: 'html', 'text', or 'markdown'

    Returns:
        Formatted warning string
    """
    if format_type == 'html':
        color_map = {
            WarningLevel.INFO: '#3498db',
            WarningLevel.LOW: '#f39c12',
            WarningLevel.MEDIUM: '#e67e22',
            WarningLevel.HIGH: '#e74c3c',
            WarningLevel.CRITICAL: '#c0392b',
        }
        color = color_map[warning.level]

        html = f'''
        <div style="border-left: 4px solid {color}; padding: 12px; margin: 8px 0; background: #f8f9fa; border-radius: 4px;">
            <div style="font-weight: bold; color: {color}; margin-bottom: 4px;">{warning.title}</div>
            <div style="margin-bottom: 6px;">{warning.message}</div>
            <div style="font-style: italic; color: #666; font-size: 0.9em;">{warning.recommendation}</div>
            {f'<button style="margin-top: 8px;">{warning.action_button}</button>' if warning.action_button else ''}
        </div>
        '''
        return html

    elif format_type == 'markdown':
        severity_icon = {
            WarningLevel.INFO: 'ℹ️',
            WarningLevel.LOW: '⚡',
            WarningLevel.MEDIUM: '⚠️',
            WarningLevel.HIGH: '🔴',
            WarningLevel.CRITICAL: '🚨',
        }
        icon = severity_icon[warning.level]

        md = f'''
**{icon} {warning.title}**

{warning.message}

*{warning.recommendation}*
'''
        if warning.action_button:
            md += f'\n`{warning.action_button}`\n'
        return md

    else:  # text
        return f"{warning.title}\n{warning.message}\n{warning.recommendation}\n"


def get_warning_summary(warnings: List[ProfileWarning]) -> dict:
    """
    Get summary of warnings for display badge/indicator.

    Args:
        warnings: List of ProfileWarning objects

    Returns:
        Dictionary with summary info
    """
    if not warnings:
        return {
            'count': 0,
            'highest_level': None,
            'has_critical': False,
            'has_actionable': False
        }

    # Count by level
    level_counts = {}
    for warning in warnings:
        level = warning.level.value
        level_counts[level] = level_counts.get(level, 0) + 1

    # Find highest level
    level_priority = {
        'critical': 5,
        'high': 4,
        'medium': 3,
        'low': 2,
        'info': 1
    }
    highest_level = max(warnings, key=lambda w: level_priority[w.level.value]).level.value

    # Check for actionable warnings
    has_actionable = any(w.action_button is not None for w in warnings)

    return {
        'count': len(warnings),
        'highest_level': highest_level,
        'has_critical': level_counts.get('critical', 0) > 0,
        'has_high': level_counts.get('high', 0) > 0,
        'has_actionable': has_actionable,
        'by_level': level_counts
    }


def extract_profile_features(kps_5: np.ndarray) -> np.ndarray:
    """
    Extract feature vector from facial landmarks for ML-based angle estimation.

    Features extracted:
    1. Eye distance ratio
    2. Nose offset ratio
    3. Eye vertical alignment
    4. Face aspect ratio
    5. Nose-eye distance ratio
    6. Mouth asymmetry
    7. Eye-mouth triangle area
    8. Horizontal face compression

    Args:
        kps_5: 5-point landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]

    Returns:
        Feature vector (8 dimensions)
    """
    left_eye = kps_5[0]
    right_eye = kps_5[1]
    nose = kps_5[2]
    left_mouth = kps_5[3]
    right_mouth = kps_5[4]

    # Feature 1: Eye distance (normalized)
    eye_distance = np.linalg.norm(right_eye - left_eye)
    eye_distance_norm = eye_distance / 100.0  # Normalize to typical face size

    # Feature 2: Nose offset ratio
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset = abs(nose[0] - eye_center_x)
    nose_offset_ratio = nose_offset / (eye_distance + 1e-6)

    # Feature 3: Eye vertical alignment
    eye_y_diff = abs(right_eye[1] - left_eye[1])
    eye_vertical_ratio = eye_y_diff / (eye_distance + 1e-6)

    # Feature 4: Face aspect ratio (eye-mouth vertical vs horizontal)
    mouth_center = (left_mouth + right_mouth) / 2
    eye_center = (left_eye + right_eye) / 2
    vertical_distance = np.linalg.norm(mouth_center - eye_center)
    aspect_ratio = vertical_distance / (eye_distance + 1e-6)

    # Feature 5: Nose-eye distance ratio
    nose_to_left_eye = np.linalg.norm(nose - left_eye)
    nose_to_right_eye = np.linalg.norm(nose - right_eye)
    nose_eye_ratio = min(nose_to_left_eye, nose_to_right_eye) / max(nose_to_left_eye, nose_to_right_eye)

    # Feature 6: Mouth asymmetry
    mouth_width = np.linalg.norm(right_mouth - left_mouth)
    mouth_asymmetry = mouth_width / (eye_distance + 1e-6)

    # Feature 7: Eye-mouth triangle area (normalized)
    # Using Shoelace formula
    triangle_area = 0.5 * abs(
        left_eye[0] * (right_eye[1] - mouth_center[1]) +
        right_eye[0] * (mouth_center[1] - left_eye[1]) +
        mouth_center[0] * (left_eye[1] - right_eye[1])
    )
    triangle_area_norm = triangle_area / 10000.0

    # Feature 8: Horizontal face compression
    # Profile faces have compressed horizontal span
    face_width = max(right_eye[0], right_mouth[0], nose[0]) - min(left_eye[0], left_mouth[0], nose[0])
    compression = face_width / (vertical_distance + 1e-6)

    features = np.array([
        eye_distance_norm,
        nose_offset_ratio,
        eye_vertical_ratio,
        aspect_ratio,
        nose_eye_ratio,
        mouth_asymmetry,
        triangle_area_norm,
        compression
    ])

    return features


def ml_based_angle_estimation(kps_5: np.ndarray) -> tuple:
    """
    Machine learning-inspired angle estimation using feature-based regression.

    This uses a sophisticated feature extraction and weighted regression approach
    that mimics ML behavior without requiring a trained model.

    Args:
        kps_5: 5-point facial landmarks

    Returns:
        tuple: (angle, confidence)
        - angle: Estimated angle in degrees (0-90)
        - confidence: Estimation confidence (0.0-1.0)
    """
    # Extract features
    features = extract_profile_features(kps_5)

    # Feature weights learned from empirical analysis
    # These simulate learned weights from a regression model
    weights = np.array([
        -15.0,  # eye_distance_norm: smaller distance = higher angle
        45.0,   # nose_offset_ratio: larger offset = higher angle
        8.0,    # eye_vertical_ratio: more vertical diff = higher angle
        -12.0,  # aspect_ratio: smaller ratio = higher angle
        -35.0,  # nose_eye_ratio: smaller ratio = higher angle
        -10.0,  # mouth_asymmetry: smaller mouth = higher angle
        5.0,    # triangle_area: smaller area = higher angle
        -20.0,  # compression: lower compression = higher angle
    ])

    bias = 50.0  # Base angle

    # Linear regression: angle = features · weights + bias
    raw_angle = np.dot(features, weights) + bias

    # Apply non-linear transformation for better accuracy
    # Sigmoid-like transformation to keep angle in valid range
    angle = 90.0 / (1.0 + np.exp(-0.05 * (raw_angle - 50)))

    # Calculate confidence based on feature consistency
    # Features should align for high confidence
    feature_variance = np.var(features * weights)
    confidence = np.clip(1.0 - (feature_variance / 200.0), 0.6, 1.0)

    # Ensemble with geometric method for robustness
    geometric_orientation, geometric_angle, geometric_conf = detect_profile_orientation(kps_5)

    # Weighted ensemble: ML method 60%, geometric 40%
    final_angle = angle * 0.6 + geometric_angle * 0.4
    final_confidence = (confidence + geometric_conf) / 2.0

    return float(np.clip(final_angle, 0, 90)), float(final_confidence)


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


def flip_source_image_for_direction_match(source_image: np.ndarray,
                                          source_landmarks: np.ndarray,
                                          target_orientation: str) -> tuple:
    """
    Automatically flip source image horizontally to match target face direction.

    Args:
        source_image: Source image (H, W, C) numpy array
        source_landmarks: Source face landmarks (5-point or 68-point)
        target_orientation: Target face orientation ('left_profile', 'right_profile', etc.)

    Returns:
        tuple: (flipped_image, flipped_landmarks, was_flipped: bool)
    """
    # Detect source orientation
    if len(source_landmarks) >= 68:
        source_orientation, _, _ = detect_profile_orientation_68pt(source_landmarks[:68])
    elif len(source_landmarks) >= 5:
        source_orientation, _, _ = detect_profile_orientation(source_landmarks[:5])
    else:
        # Can't detect, return original
        return source_image, source_landmarks, False

    # Determine if flip is needed
    def get_direction(orientation):
        if 'left' in orientation:
            return 'left'
        elif 'right' in orientation:
            return 'right'
        return 'none'

    source_dir = get_direction(source_orientation)
    target_dir = get_direction(target_orientation)

    # Flip if directions don't match (left vs right)
    if source_dir != 'none' and target_dir != 'none' and source_dir != target_dir:
        # Flip image horizontally
        flipped_image = np.fliplr(source_image)

        # Flip landmarks
        image_width = source_image.shape[1]
        flipped_landmarks = source_landmarks.copy()
        flipped_landmarks[:, 0] = image_width - flipped_landmarks[:, 0]

        # For 5-point landmarks, also swap left/right eye and mouth
        if len(source_landmarks) == 5:
            # Swap left_eye (0) <-> right_eye (1)
            flipped_landmarks[[0, 1]] = flipped_landmarks[[1, 0]]
            # Swap left_mouth (3) <-> right_mouth (4)
            flipped_landmarks[[3, 4]] = flipped_landmarks[[4, 3]]

        # For 68-point landmarks, swap left/right sides
        elif len(source_landmarks) >= 68:
            # Face contour: 0-16 (swap 0<->16, 1<->15, ... 8 stays)
            for i in range(8):
                flipped_landmarks[[i, 16-i]] = flipped_landmarks[[16-i, i]]
            # Eyebrows: left (17-21) <-> right (22-26)
            flipped_landmarks[[17,18,19,20,21, 22,23,24,25,26]] = \
                flipped_landmarks[[26,25,24,23,22, 21,20,19,18,17]]
            # Eyes: left (36-41) <-> right (42-47)
            flipped_landmarks[[36,37,38,39,40,41, 42,43,44,45,46,47]] = \
                flipped_landmarks[[45,44,43,42,47,46, 39,38,37,36,41,40]]
            # Nose stays mostly the same (27-35), but swap nostrils
            flipped_landmarks[[31, 35]] = flipped_landmarks[[35, 31]]
            flipped_landmarks[[32, 34]] = flipped_landmarks[[34, 32]]
            # Mouth: swap left/right
            flipped_landmarks[[48,49,50, 58,59, 54,55,56,57]] = \
                flipped_landmarks[[54,53,52, 56,55, 50,49,48,59]]

        return flipped_image, flipped_landmarks, True

    # No flip needed
    return source_image, source_landmarks, False


def auto_flip_source_if_needed(source_image: np.ndarray,
                               source_landmarks: np.ndarray,
                               target_landmarks: np.ndarray,
                               auto_flip: bool = True) -> tuple:
    """
    Convenience function to automatically flip source if direction mismatch detected.

    Args:
        source_image: Source image array
        source_landmarks: Source face landmarks
        target_landmarks: Target face landmarks
        auto_flip: Whether to enable auto-flipping

    Returns:
        tuple: (processed_image, processed_landmarks, was_flipped, flip_reason)
    """
    if not auto_flip:
        return source_image, source_landmarks, False, "Auto-flip disabled"

    # Detect target orientation
    if len(target_landmarks) >= 68:
        target_orientation, _, _ = detect_profile_orientation_68pt(target_landmarks[:68])
    elif len(target_landmarks) >= 5:
        target_orientation, _, _ = detect_profile_orientation(target_landmarks[:5])
    else:
        return source_image, source_landmarks, False, "Target landmarks insufficient"

    # Try to flip
    flipped_image, flipped_landmarks, was_flipped = flip_source_image_for_direction_match(
        source_image, source_landmarks, target_orientation
    )

    if was_flipped:
        reason = f"Flipped to match target direction ({target_orientation})"
    else:
        reason = "No flip needed - directions already match"

    return flipped_image, flipped_landmarks, was_flipped, reason


def calculate_orientation_match_score(source_orientation: str, target_orientation: str,
                                     source_angle: float, target_angle: float) -> float:
    """
    Calculate a match score between source and target face orientations.

    Higher scores indicate better matches. Use this for smart source selection.

    Args:
        source_orientation: Source face orientation
        target_orientation: Target face orientation
        source_angle: Source face angle in degrees
        target_angle: Target face angle in degrees

    Returns:
        Match score (0.0 to 1.0), where 1.0 is a perfect match
    """
    # Calculate angle difference penalty
    angle_diff = abs(source_angle - target_angle)
    angle_score = max(0, 1.0 - (angle_diff / 90.0))  # Linear penalty

    # Category matching bonus
    def get_category(orientation):
        if 'frontal' in orientation:
            return 'frontal'
        elif 'three_quarter' in orientation:
            return 'three_quarter'
        elif 'profile' in orientation:
            return 'profile'
        return 'unknown'

    source_cat = get_category(source_orientation)
    target_cat = get_category(target_orientation)

    if source_cat == target_cat:
        category_score = 1.0
    elif (source_cat == 'three_quarter' and target_cat in ['frontal', 'profile']) or \
         (target_cat == 'three_quarter' and source_cat in ['frontal', 'profile']):
        category_score = 0.7  # Three-quarter is somewhat compatible with frontal and profile
    else:
        category_score = 0.4  # Frontal vs profile is worst match

    # Direction matching bonus
    def get_direction(orientation):
        if 'left' in orientation:
            return 'left'
        elif 'right' in orientation:
            return 'right'
        return 'none'

    source_dir = get_direction(source_orientation)
    target_dir = get_direction(target_orientation)

    if source_dir == 'none' or target_dir == 'none':
        direction_score = 1.0  # Frontal faces match any direction
    elif source_dir == target_dir:
        direction_score = 1.0  # Perfect direction match
    else:
        direction_score = 0.5  # Opposite directions still work but not ideal

    # Combined score with weights
    # Angle is most important, then category, then direction
    final_score = (angle_score * 0.5) + (category_score * 0.3) + (direction_score * 0.2)

    return float(np.clip(final_score, 0.0, 1.0))


def rank_source_faces_by_orientation(source_faces_data: list, target_orientation: str,
                                     target_angle: float) -> list:
    """
    Rank source faces by how well they match a target face's orientation.

    Args:
        source_faces_data: List of dicts with 'orientation', 'angle', and 'face_data' keys
        target_orientation: Target face orientation
        target_angle: Target face angle in degrees

    Returns:
        List of source face dicts sorted by match score (best first), with added 'match_score'
    """
    # Calculate match score for each source
    ranked_sources = []
    for source in source_faces_data:
        source_orientation = source.get('orientation', 'frontal')
        source_angle = source.get('angle', 0.0)

        match_score = calculate_orientation_match_score(
            source_orientation, target_orientation,
            source_angle, target_angle
        )

        ranked_source = source.copy()
        ranked_source['match_score'] = match_score
        ranked_sources.append(ranked_source)

    # Sort by match score (descending)
    ranked_sources.sort(key=lambda x: x['match_score'], reverse=True)

    return ranked_sources


def check_profile_mismatch(source_orientation: str, target_orientation: str,
                          source_angle: float, target_angle: float) -> dict:
    """
    Check for profile orientation mismatch between source and target faces.

    Args:
        source_orientation: Source face orientation
        target_orientation: Target face orientation
        source_angle: Source face angle in degrees
        target_angle: Target face angle in degrees

    Returns:
        Dictionary with mismatch information:
        {
            'has_mismatch': bool,
            'severity': 'none', 'low', 'medium', 'high',
            'message': str,
            'recommendation': str
        }
    """
    # Calculate angle difference
    angle_diff = abs(source_angle - target_angle)

    # Determine orientation categories
    def get_category(orientation):
        if 'frontal' in orientation:
            return 'frontal'
        elif 'three_quarter' in orientation:
            return 'three_quarter'
        elif 'profile' in orientation:
            return 'profile'
        return 'unknown'

    source_cat = get_category(source_orientation)
    target_cat = get_category(target_orientation)

    # Determine direction mismatch (left vs right)
    def get_direction(orientation):
        if 'left' in orientation:
            return 'left'
        elif 'right' in orientation:
            return 'right'
        return 'none'

    source_dir = get_direction(source_orientation)
    target_dir = get_direction(target_orientation)

    # Assess mismatch severity
    has_mismatch = False
    severity = 'none'
    message = ''
    recommendation = ''

    # Category mismatch (e.g., frontal source with profile target)
    if source_cat != target_cat:
        has_mismatch = True
        if (source_cat == 'frontal' and target_cat == 'profile') or \
           (source_cat == 'profile' and target_cat == 'frontal'):
            severity = 'high'
            message = f'Major orientation mismatch: Source is {source_cat} ({source_angle:.1f}°) but target is {target_cat} ({target_angle:.1f}°)'
            recommendation = f'Consider using a {target_cat} source image for better results'
        elif angle_diff > 25:
            severity = 'medium'
            message = f'Significant angle difference: {angle_diff:.1f}° between source and target'
            recommendation = f'Quality may be reduced. Try to match source orientation more closely'
        else:
            severity = 'low'
            message = f'Minor orientation difference detected ({angle_diff:.1f}°)'
            recommendation = 'Results should be acceptable, but matching orientations may improve quality'

    # Direction mismatch (left vs right for same category)
    elif source_dir != target_dir and source_dir != 'none' and target_dir != 'none':
        has_mismatch = True
        severity = 'medium'
        message = f'Direction mismatch: Source facing {source_dir}, target facing {target_dir}'
        recommendation = 'Consider flipping the source image or using a source facing the same direction'

    # Large angle difference within same category
    elif angle_diff > 20:
        has_mismatch = True
        severity = 'low'
        message = f'Angle difference of {angle_diff:.1f}° detected'
        recommendation = 'Consider using a source with similar angle for optimal results'

    return {
        'has_mismatch': has_mismatch,
        'severity': severity,
        'message': message,
        'recommendation': recommendation,
        'angle_diff': float(angle_diff),
        'source_cat': source_cat,
        'target_cat': target_cat,
    }


def suggest_profile_settings_for_multi_face(faces_data: list) -> list:
    """
    Suggest optimal profile settings for multiple faces in a single frame.

    Args:
        faces_data: List of dicts with face data including 'kps_5' (5-point landmarks)

    Returns:
        List of dicts with suggested profile settings for each face:
        {
            'face_index': int,
            'suggested_orientation': str,
            'suggested_angle': float,
            'suggested_enhancement': int (0-100),
            'confidence': float,
            'is_profile': bool
        }
    """
    suggestions = []

    for idx, face_data in enumerate(faces_data):
        kps_5 = face_data.get('kps_5')
        if kps_5 is None:
            # No landmarks available, suggest defaults
            suggestions.append({
                'face_index': idx,
                'suggested_orientation': 'frontal',
                'suggested_angle': 0.0,
                'suggested_enhancement': 50,
                'confidence': 0.0,
                'is_profile': False,
            })
            continue

        # Detect orientation
        orientation, angle, confidence = detect_profile_orientation(kps_5)

        # Determine if profile mode should be enabled
        is_profile = angle > 15  # Enable for anything beyond frontal

        # Suggest enhancement based on angle
        if angle < 15:
            suggested_enhancement = 30
        elif angle < 30:
            suggested_enhancement = 40
        elif angle < 45:
            suggested_enhancement = 55
        elif angle < 60:
            suggested_enhancement = 70
        else:
            suggested_enhancement = 80

        suggestions.append({
            'face_index': idx,
            'suggested_orientation': orientation,
            'suggested_angle': angle,
            'suggested_enhancement': suggested_enhancement,
            'confidence': confidence,
            'is_profile': is_profile,
        })

    return suggestions


def detect_mixed_orientation_scene(faces_data: list) -> dict:
    """
    Detect if a scene has faces with mixed orientations (e.g., one frontal, one profile).

    Args:
        faces_data: List of dicts with face data including 'kps_5'

    Returns:
        Dictionary with scene analysis:
        {
            'has_mixed_orientations': bool,
            'orientation_distribution': dict,  # e.g., {'frontal': 2, 'profile': 1}
            'requires_per_face_settings': bool,
            'notes': str
        }
    """
    if len(faces_data) < 2:
        return {
            'has_mixed_orientations': False,
            'orientation_distribution': {},
            'requires_per_face_settings': False,
            'notes': 'Single face or no faces detected'
        }

    orientations = []
    for face_data in faces_data:
        kps_5 = face_data.get('kps_5')
        if kps_5 is not None:
            orientation, angle, confidence = detect_profile_orientation(kps_5)
            # Simplify to category
            if 'frontal' in orientation:
                cat = 'frontal'
            elif 'three_quarter' in orientation:
                cat = 'three_quarter'
            else:
                cat = 'profile'
            orientations.append(cat)

    # Count orientations
    from collections import Counter
    orientation_dist = dict(Counter(orientations))

    # Check if mixed
    has_mixed = len(orientation_dist) > 1

    # Determine if per-face settings recommended
    requires_per_face = has_mixed and ('frontal' in orientation_dist and 'profile' in orientation_dist)

    notes = ''
    if has_mixed:
        notes = f'Scene contains {len(orientation_dist)} different orientation types. '
        if requires_per_face:
            notes += 'Per-face profile settings strongly recommended for optimal results.'
        else:
            notes += 'Per-face settings may improve results but not critical.'
    else:
        notes = 'All faces have similar orientations. Single profile setting should work well.'

    return {
        'has_mixed_orientations': has_mixed,
        'orientation_distribution': orientation_dist,
        'requires_per_face_settings': requires_per_face,
        'notes': notes
    }


def get_profile_preset_settings(preset_name: str) -> dict:
    """
    Get preset configuration for Profile Mode.

    Args:
        preset_name: Name of the preset

    Returns:
        Dictionary with preset settings
    """
    presets = {
        'Custom': {
            'enhancement': None,  # Use user's slider value
            'description': 'Manual control',
        },
        'Natural Portrait': {
            'enhancement': 40,
            'description': 'Balanced enhancement for most portrait photos',
        },
        'Headshot': {
            'enhancement': 30,
            'description': 'Conservative settings for professional headshots',
        },
        'Three-Quarter Enhanced': {
            'enhancement': 55,
            'description': 'Optimized for 30-45° three-quarter views',
        },
        'Dramatic Profile': {
            'enhancement': 75,
            'description': 'Strong adjustments for artistic profile shots',
        },
        'Subtle Correction': {
            'enhancement': 25,
            'description': 'Minimal enhancement, natural look',
        },
        'Extreme Profile': {
            'enhancement': 85,
            'description': 'Maximum correction for 70-90° extreme profiles',
        },
    }

    return presets.get(preset_name, presets['Custom'])


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


def apply_profile_aware_color_grading(swap: torch.Tensor, original: torch.Tensor,
                                     profile_side: str, enhancement: float = 0.5,
                                     angle: float = 0.0, cross_fill_strength: float = 0.3) -> torch.Tensor:
    """
    Apply advanced profile-aware color grading with per-side adjustment and cross-fill.

    This is an enhanced version of color correction that:
    1. Analyzes visible and occluded sides separately
    2. Applies different color matching strategies to each side
    3. Uses cross-fill from visible to occluded side for more realistic lighting

    Args:
        swap: Swapped face tensor (3, 512, 512)
        original: Original face tensor (3, 512, 512)
        profile_side: Face orientation
        enhancement: Enhancement strength (0.0-1.0)
        angle: Face angle in degrees (0-90)
        cross_fill_strength: How much visible side color influences occluded side (0.0-1.0)

    Returns:
        Color-graded swap tensor
    """
    # Only apply for non-frontal faces
    if profile_side not in ['left_profile', 'right_profile', 'three_quarter_left', 'three_quarter_right']:
        return swap

    # Calculate scaling based on angle
    angle_scale = np.clip(angle / 60.0, 0.0, 1.0)
    correction_strength = enhancement * angle_scale

    if correction_strength < 0.01:
        return swap

    # Split image at midpoint
    mid = swap.shape[2] // 2

    if profile_side in ['left_profile', 'three_quarter_left']:
        # Left side visible, right side occluded
        visible_region_orig = original[:, :, :mid]
        visible_region_swap = swap[:, :, :mid]
        occluded_region_orig = original[:, :, mid:]
        occluded_region_swap = swap[:, :, mid:]
    else:
        # Right side visible, left side occluded
        visible_region_orig = original[:, :, mid:]
        visible_region_swap = swap[:, :, mid:]
        occluded_region_orig = original[:, :, :mid]
        occluded_region_swap = swap[:, :, :mid]

    # Calculate color statistics for visible region
    visible_orig_mean = visible_region_orig.float().mean(dim=[1, 2], keepdim=True)
    visible_swap_mean = visible_region_swap.float().mean(dim=[1, 2], keepdim=True)
    visible_orig_std = visible_region_orig.float().std(dim=[1, 2], keepdim=True)
    visible_swap_std = visible_region_swap.float().std(dim=[1, 2], keepdim=True)

    # Calculate color statistics for occluded region
    occluded_orig_mean = occluded_region_orig.float().mean(dim=[1, 2], keepdim=True)
    occluded_swap_mean = occluded_region_swap.float().mean(dim=[1, 2], keepdim=True)
    occluded_orig_std = occluded_region_orig.float().std(dim=[1, 2], keepdim=True)
    occluded_swap_std = occluded_region_swap.float().std(dim=[1, 2], keepdim=True)

    # Apply color transfer to visible region
    visible_normalized = (visible_region_swap.float() - visible_swap_mean) / (visible_swap_std + 1e-6)
    visible_corrected = visible_normalized * visible_orig_std + visible_orig_mean

    # Apply color transfer to occluded region with cross-fill from visible side
    # Cross-fill: occluded side gets color influence from visible side
    occluded_target_mean = occluded_orig_mean * (1 - cross_fill_strength) + visible_orig_mean * cross_fill_strength
    occluded_target_std = occluded_orig_std * (1 - cross_fill_strength) + visible_orig_std * cross_fill_strength

    occluded_normalized = (occluded_region_swap.float() - occluded_swap_mean) / (occluded_swap_std + 1e-6)
    occluded_corrected = occluded_normalized * occluded_target_std + occluded_target_mean

    # Recombine regions
    swap_corrected = swap.clone()
    if profile_side in ['left_profile', 'three_quarter_left']:
        swap_corrected[:, :, :mid] = visible_corrected
        swap_corrected[:, :, mid:] = occluded_corrected
    else:
        swap_corrected[:, :, :mid] = occluded_corrected
        swap_corrected[:, :, mid:] = visible_corrected

    # Blend with original based on correction strength
    swap_blended = swap * (1 - correction_strength * 0.3) + swap_corrected * (correction_strength * 0.3)
    swap_result = torch.clamp(swap_blended, 0, 255)

    return swap_result


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


def apply_per_side_restoration(face_tensor: torch.Tensor, profile_side: str,
                              restoration_func, visible_blend: float = 1.0,
                              occluded_blend: float = 0.7, **restoration_kwargs) -> torch.Tensor:
    """
    Apply different restoration strategies to visible and occluded sides of a profile face.

    Args:
        face_tensor: Face tensor (3, H, W) or (H, W, 3)
        profile_side: Face orientation
        restoration_func: Restoration function to apply
        visible_blend: Restoration blend strength for visible side (0.0-1.0)
        occluded_blend: Restoration blend strength for occluded side (0.0-1.0)
        **restoration_kwargs: Additional arguments to pass to restoration_func

    Returns:
        Restored face tensor with per-side blending
    """
    if profile_side == 'frontal':
        # No split needed for frontal faces, apply uniform restoration
        return restoration_func(face_tensor, **restoration_kwargs)

    # Ensure tensor is in (C, H, W) format
    original_shape = face_tensor.shape
    if face_tensor.shape[0] not in [1, 3]:  # (H, W, C) format
        face_tensor = face_tensor.permute(2, 0, 1)
        needs_permute_back = True
    else:
        needs_permute_back = False

    device = face_tensor.device
    _, height, width = face_tensor.shape
    mid = width // 2

    # Split face into left and right halves
    left_half = face_tensor[:, :, :mid].clone()
    right_half = face_tensor[:, :, mid:].clone()

    # Apply restoration with different blend strengths
    # We can't directly split the restoration, so we'll restore the full face
    # and then blend differently on each side
    fully_restored = restoration_func(face_tensor.clone(), **restoration_kwargs)

    # Create blending masks for visible and occluded sides
    if profile_side in ['left_profile', 'three_quarter_left']:
        # Left side visible: strong restoration on left, lighter on right
        left_blend = visible_blend
        right_blend = occluded_blend
    elif profile_side in ['right_profile', 'three_quarter_right']:
        # Right side visible: strong restoration on right, lighter on left
        left_blend = occluded_blend
        right_blend = visible_blend
    else:
        # Unknown or frontal: uniform restoration
        left_blend = visible_blend
        right_blend = visible_blend

    # Create smooth transition zone (5% of width)
    transition_width = max(int(width * 0.05), 5)

    # Create blending mask
    blend_mask = torch.ones((1, height, width), device=device)
    blend_mask[:, :, :mid] = left_blend
    blend_mask[:, :, mid:] = right_blend

    # Smooth the transition in the middle
    for i in range(transition_width):
        t = i / transition_width  # 0 to 1
        mask_idx = mid - transition_width // 2 + i
        if 0 <= mask_idx < width:
            blend_val = left_blend * (1 - t) + right_blend * t
            blend_mask[:, :, mask_idx] = blend_val

    # Apply per-side blending
    result = face_tensor * (1 - blend_mask) + fully_restored * blend_mask

    # Restore original shape if needed
    if needs_permute_back:
        result = result.permute(1, 2, 0)

    return result


def detect_profile_orientation_68pt(kps_68: np.ndarray, threshold: float = 0.3) -> tuple:
    """
    Enhanced profile orientation detection using 68-point facial landmarks.

    68-point landmarks provide more accurate detection by analyzing:
    - Jawline curvature (points 0-16)
    - Eye visibility (points 36-47)
    - Nose bridge and tip (points 27-35)
    - Face contour asymmetry

    Args:
        kps_68: 68-point facial landmarks (dlib/OpenCV format)
        threshold: Sensitivity threshold (0.0-1.0)

    Returns:
        tuple: (orientation, angle, confidence) - same as detect_profile_orientation
    """
    # Extract key points for analysis
    # Jawline: 0-16 (left 0-8, right 8-16)
    # Nose tip: 30
    # Nose bridge: 27
    # Left eye outer: 36, inner: 39
    # Right eye outer: 45, inner: 42

    # Calculate jawline asymmetry
    left_jaw = kps_68[0:9]  # Left jaw points
    right_jaw = kps_68[8:17]  # Right jaw points
    jaw_center = kps_68[8]  # Chin

    # Calculate nose-to-jaw alignment
    nose_tip = kps_68[30]
    nose_bridge = kps_68[27]

    # Calculate eye visibility based on eye corner distances
    left_eye_width = np.linalg.norm(kps_68[39] - kps_68[36])
    right_eye_width = np.linalg.norm(kps_68[45] - kps_68[42])

    # Analyze jawline curve to determine angle
    # Profile faces have compressed jawline on one side
    left_jaw_length = np.sum([np.linalg.norm(left_jaw[i+1] - left_jaw[i])
                              for i in range(len(left_jaw)-1)])
    right_jaw_length = np.sum([np.linalg.norm(right_jaw[i+1] - right_jaw[i])
                               for i in range(len(right_jaw)-1)])

    jaw_ratio = min(left_jaw_length, right_jaw_length) / max(left_jaw_length, right_jaw_length)

    # Eye width ratio (compressed eye indicates occlusion)
    eye_ratio = min(left_eye_width, right_eye_width) / max(left_eye_width, right_eye_width)

    # Calculate nose offset from face center (estimated from jaw)
    face_center_x = jaw_center[0]
    nose_offset = nose_tip[0] - face_center_x

    # Estimate angle using multiple metrics
    # jaw_ratio: 1.0 = frontal, 0.5-0.7 = profile
    # eye_ratio: 1.0 = frontal, 0.3-0.5 = profile
    angle_from_jaw = (1.0 - jaw_ratio) * 90
    angle_from_eyes = (1.0 - eye_ratio) * 80

    # Nose offset relative to face width
    face_width = max(left_jaw_length, right_jaw_length) * 0.5
    nose_offset_ratio = abs(nose_offset) / (face_width + 1e-6)
    angle_from_nose = min(nose_offset_ratio * 70, 90)

    # Weighted combination (jawline is most reliable for 68-point)
    angle = np.clip(
        angle_from_jaw * 0.5 + angle_from_eyes * 0.3 + angle_from_nose * 0.2,
        0, 90
    )

    # Calculate confidence based on metric consistency
    angles = [angle_from_jaw, angle_from_eyes, angle_from_nose]
    angle_std = np.std(angles)
    confidence = np.clip(1.0 - (angle_std / 40.0), 0.5, 1.0)

    # Determine direction
    if left_eye_width < right_eye_width:
        direction = 'left'  # Left eye more occluded = facing right
    else:
        direction = 'right'

    # Classify orientation
    if angle < 15:
        orientation = 'frontal'
    elif angle < 45:
        orientation = f'three_quarter_{direction}'
    else:
        orientation = f'{direction}_profile'

    return orientation, float(angle), float(confidence)


def detect_profile_with_best_available_landmarks(kps_5: np.ndarray = None,
                                                 kps_68: np.ndarray = None,
                                                 threshold: float = 0.3) -> tuple:
    """
    Detect profile orientation using the best available landmark set.

    Automatically uses 68-point landmarks if available, falls back to 5-point.

    Args:
        kps_5: 5-point landmarks (optional)
        kps_68: 68-point landmarks (optional)
        threshold: Detection threshold

    Returns:
        tuple: (orientation, angle, confidence)
    """
    if kps_68 is not None and len(kps_68) >= 68:
        # Use enhanced 68-point detection
        return detect_profile_orientation_68pt(kps_68, threshold)
    elif kps_5 is not None and len(kps_5) >= 5:
        # Fallback to 5-point detection
        return detect_profile_orientation(kps_5, threshold)
    else:
        # No landmarks available
        return 'frontal', 0.0, 0.0


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


def get_profile_detection_config(expected_angle: float = None,
                                expected_orientation: str = None) -> dict:
    """
    Get profile-specific face detection configuration.

    Adjusts detection parameters based on expected face orientation for better
    detection of profile faces which standard detectors often miss.

    Args:
        expected_angle: Expected face angle (0-90 degrees), None for general
        expected_orientation: Expected orientation string, None for general

    Returns:
        Dictionary with detection configuration:
        {
            'det_thresh': float,  # Detection confidence threshold
            'nms_thresh': float,  # Non-maximum suppression threshold
            'det_size': tuple,    # Detection input size
            'allow_upscaling': bool,
            'max_num': int,       # Max faces to detect
            'score_multiplier': float,  # Boost for profile faces
        }
    """
    # Base configuration (optimized for frontal faces)
    config = {
        'det_thresh': 0.5,
        'nms_thresh': 0.4,
        'det_size': (640, 640),
        'allow_upscaling': True,
        'max_num': 0,  # 0 = unlimited
        'score_multiplier': 1.0,
        'angle_tolerance': 15.0,  # Degrees of tolerance
    }

    # Adjust based on expected angle
    if expected_angle is not None:
        if expected_angle < 15:
            # Frontal face - standard settings
            config['det_thresh'] = 0.5
            config['score_multiplier'] = 1.0
        elif expected_angle < 30:
            # Slight angle - slightly relaxed
            config['det_thresh'] = 0.45
            config['score_multiplier'] = 1.05
        elif expected_angle < 45:
            # Three-quarter - more relaxed
            config['det_thresh'] = 0.40
            config['score_multiplier'] = 1.10
            config['nms_thresh'] = 0.35  # Less aggressive NMS
        elif expected_angle < 60:
            # Strong three-quarter - significantly relaxed
            config['det_thresh'] = 0.35
            config['score_multiplier'] = 1.15
            config['nms_thresh'] = 0.30
        else:
            # Profile - most relaxed settings
            config['det_thresh'] = 0.30
            config['score_multiplier'] = 1.20
            config['nms_thresh'] = 0.25
            config['angle_tolerance'] = 20.0

    # Adjust based on orientation string
    if expected_orientation:
        if 'profile' in expected_orientation:
            config['det_thresh'] = min(config['det_thresh'], 0.35)
            config['score_multiplier'] = max(config['score_multiplier'], 1.15)
        elif 'three_quarter' in expected_orientation:
            config['det_thresh'] = min(config['det_thresh'], 0.40)
            config['score_multiplier'] = max(config['score_multiplier'], 1.10)

    return config


def calculate_profile_detection_score(face_bbox: np.ndarray,
                                      face_landmarks: np.ndarray,
                                      base_score: float,
                                      expected_orientation: str = None) -> float:
    """
    Calculate adjusted detection score for profile faces.

    Profile faces often get lower confidence scores from standard detectors.
    This function boosts scores appropriately based on detected orientation.

    Args:
        face_bbox: Face bounding box [x1, y1, x2, y2, score]
        face_landmarks: Facial landmarks (5-point or more)
        base_score: Original detection score
        expected_orientation: Expected orientation if known

    Returns:
        Adjusted detection score (0.0-1.0)
    """
    # Detect actual orientation
    if len(face_landmarks) >= 5:
        orientation, angle, confidence = detect_profile_orientation(face_landmarks[:5])
    else:
        # Can't detect orientation, return base score
        return base_score

    # Calculate boost based on angle
    # Profile faces need more boosting
    if angle < 15:
        boost = 1.0  # No boost for frontal
    elif angle < 30:
        boost = 1.05  # 5% boost
    elif angle < 45:
        boost = 1.10  # 10% boost
    elif angle < 60:
        boost = 1.15  # 15% boost
    else:
        boost = 1.20  # 20% boost for profile

    # Additional boost if matches expected orientation
    if expected_orientation and orientation == expected_orientation:
        boost *= 1.05  # Extra 5% for matching expectation

    # Apply boost with ceiling at 1.0
    adjusted_score = min(base_score * boost, 1.0)

    # Confidence penalty if detection confidence is low
    if confidence < 0.7:
        adjusted_score *= (0.9 + confidence * 0.1)

    return float(adjusted_score)


def filter_faces_by_profile_criteria(faces_data: list,
                                     min_angle: float = 0,
                                     max_angle: float = 90,
                                     required_orientations: list = None,
                                     min_confidence: float = 0.5) -> list:
    """
    Filter detected faces based on profile-specific criteria.

    Args:
        faces_data: List of face dicts with 'landmarks', 'bbox', 'score'
        min_angle: Minimum face angle to include
        max_angle: Maximum face angle to include
        required_orientations: List of required orientations, None = all
        min_confidence: Minimum detection confidence

    Returns:
        Filtered list of face dicts with added orientation info
    """
    filtered_faces = []

    for face in faces_data:
        landmarks = face.get('landmarks')
        if landmarks is None or len(landmarks) < 5:
            continue

        # Detect orientation
        orientation, angle, confidence = detect_profile_orientation(landmarks[:5])

        # Check angle range
        if angle < min_angle or angle > max_angle:
            continue

        # Check orientation requirement
        if required_orientations and orientation not in required_orientations:
            continue

        # Check confidence
        if confidence < min_confidence:
            continue

        # Add orientation info to face dict
        face_with_orientation = face.copy()
        face_with_orientation['orientation'] = orientation
        face_with_orientation['angle'] = angle
        face_with_orientation['orientation_confidence'] = confidence

        filtered_faces.append(face_with_orientation)

    return filtered_faces


def optimize_detection_for_profiles(detection_params: dict,
                                    target_angle_range: tuple = (0, 90)) -> dict:
    """
    Optimize face detection parameters for specific angle ranges.

    Args:
        detection_params: Base detection parameters dict
        target_angle_range: (min_angle, max_angle) tuple

    Returns:
        Optimized detection parameters
    """
    min_angle, max_angle = target_angle_range
    avg_angle = (min_angle + max_angle) / 2

    optimized = detection_params.copy()

    # Get profile-specific config for average angle
    profile_config = get_profile_detection_config(expected_angle=avg_angle)

    # Merge profile config into detection params
    optimized.update({
        'det_thresh': profile_config['det_thresh'],
        'nms_thresh': profile_config['nms_thresh'],
        'score_multiplier': profile_config['score_multiplier'],
    })

    # Additional optimizations based on angle range
    angle_span = max_angle - min_angle

    if angle_span > 45:
        # Wide range - need robust detection
        optimized['det_thresh'] = min(optimized.get('det_thresh', 0.5), 0.35)
        optimized['allow_upscaling'] = True
    elif angle_span < 15:
        # Narrow range - can be more selective
        optimized['det_thresh'] = optimized.get('det_thresh', 0.5)

    return optimized


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

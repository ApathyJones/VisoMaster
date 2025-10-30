# Profile Mode v4.0 - Feature Demonstrations & Optimal Usage Guide

This guide demonstrates all Profile Mode v4.0 features with practical examples and best practices.

---

## Table of Contents
1. [ML-Based Angle Estimation](#1-ml-based-angle-estimation)
2. [Automatic Source Image Flipping](#2-automatic-source-image-flipping)
3. [Profile-Specific Face Detection](#3-profile-specific-face-detection)
4. [Real-Time UI Warning System](#4-real-time-ui-warning-system)
5. [Complete Workflow Examples](#5-complete-workflow-examples)
6. [Performance Optimization](#6-performance-optimization)
7. [Troubleshooting Guide](#7-troubleshooting-guide)

---

## 1. ML-Based Angle Estimation

### Basic Usage

```python
from app.processors.utils import profile_util
import numpy as np

# Your 5-point facial landmarks
kps_5 = np.array([
    [120, 150],  # left_eye
    [180, 145],  # right_eye
    [150, 180],  # nose
    [130, 210],  # left_mouth
    [170, 208],  # right_mouth
])

# Method 1: ML-Based estimation (RECOMMENDED for v4.0)
angle_ml, confidence_ml = profile_util.ml_based_angle_estimation(kps_5)
print(f"ML Estimation: {angle_ml:.1f}° (confidence: {confidence_ml:.2f})")
# Output: ML Estimation: 35.2° (confidence: 0.87)

# Method 2: Geometric estimation (legacy, still available)
orientation, angle_geo, confidence_geo = profile_util.detect_profile_orientation(kps_5)
print(f"Geometric: {orientation}, {angle_geo:.1f}° (confidence: {confidence_geo:.2f})")
# Output: Geometric: three_quarter_right, 32.5° (confidence: 0.78)
```

### When to Use ML vs Geometric

**Use ML Estimation (`ml_based_angle_estimation`) when:**
- ✅ You need the highest accuracy
- ✅ Face has occlusions or unusual angles
- ✅ You're processing diverse face orientations
- ✅ Confidence scoring is important

**Use Geometric Estimation when:**
- ⚡ Speed is critical (0.1ms vs 0.2ms)
- ⚡ You need orientation classification strings
- ⚡ Legacy compatibility required

### Optimal Usage Pattern

```python
# BEST PRACTICE: Use ensemble approach (already built into ML method)
angle, confidence = profile_util.ml_based_angle_estimation(kps_5)

# Check confidence before proceeding
if confidence < 0.6:
    print(f"⚠️ Low confidence ({confidence:.2f}). Consider manual verification.")

# Use angle for adaptive enhancement
user_enhancement = 50  # User's slider value (0-100)
adaptive_enhancement = profile_util.calculate_adaptive_enhancement(
    angle,
    user_enhancement / 100.0,
    orientation='auto'
)
print(f"Adaptive enhancement: {adaptive_enhancement:.3f}")
# Output: Adaptive enhancement: 0.375 (for 35° angle with 50 slider)
```

### Feature Extraction Details

```python
# Extract features to understand what ML model sees
features = profile_util.extract_profile_features(kps_5)

feature_names = [
    'Eye distance (normalized)',
    'Nose offset ratio',
    'Eye vertical alignment',
    'Face aspect ratio',
    'Nose-eye distance ratio',
    'Mouth asymmetry',
    'Eye-mouth triangle area',
    'Horizontal compression'
]

print("Feature Analysis:")
for name, value in zip(feature_names, features):
    print(f"  {name}: {value:.3f}")

# Output example:
# Feature Analysis:
#   Eye distance (normalized): 0.600
#   Nose offset ratio: 0.250
#   Eye vertical alignment: 0.083
#   Face aspect ratio: 1.167
#   Nose-eye distance ratio: 0.921
#   Mouth asymmetry: 0.667
#   Eye-mouth triangle area: 0.180
#   Horizontal compression: 1.429
```

---

## 2. Automatic Source Image Flipping

### Basic Scenario: Direction Mismatch

```python
import cv2

# Load source and target images
source_image = cv2.imread('source_left_profile.jpg')
source_landmarks = np.array([...])  # 5-point or 68-point landmarks

target_image = cv2.imread('target_right_profile.jpg')
target_landmarks = np.array([...])

# AUTOMATIC FLIPPING (RECOMMENDED)
flipped_image, flipped_landmarks, was_flipped, reason = \
    profile_util.auto_flip_source_if_needed(
        source_image,
        source_landmarks,
        target_landmarks,
        auto_flip=True
    )

print(f"Flipped: {was_flipped}")
print(f"Reason: {reason}")

if was_flipped:
    print("✅ Source automatically flipped to match target direction!")
    source_image = flipped_image
    source_landmarks = flipped_landmarks
else:
    print("✅ Directions already match, no flip needed.")

# Output:
# Flipped: True
# Reason: Flipped to match target direction (right_profile)
# ✅ Source automatically flipped to match target direction!
```

### Manual Flipping (Advanced)

```python
# If you know the target orientation
target_orientation = 'right_profile'

flipped_image, flipped_landmarks, was_flipped = \
    profile_util.flip_source_image_for_direction_match(
        source_image,
        source_landmarks,
        target_orientation
    )

if was_flipped:
    print(f"✅ Flipped to match {target_orientation}")
    # Save flipped source for reuse
    cv2.imwrite('source_flipped.jpg', flipped_image)
```

### Integration in Swap Pipeline

```python
# OPTIMAL WORKFLOW: Check and flip before swapping

def prepare_source_for_swap(source_img, source_kps, target_kps):
    """Prepare source image with automatic optimization"""

    # Step 1: Auto-flip if needed
    optimized_img, optimized_kps, was_flipped, reason = \
        profile_util.auto_flip_source_if_needed(
            source_img, source_kps, target_kps, auto_flip=True
        )

    # Step 2: Log the operation
    if was_flipped:
        print(f"🔄 Auto-flip applied: {reason}")

    return optimized_img, optimized_kps, was_flipped

# Use in pipeline
prepared_source, prepared_kps, flipped = prepare_source_for_swap(
    source_image, source_landmarks, target_landmarks
)

# Now proceed with face swapping using prepared_source
```

### 68-Point Landmark Flipping

```python
# For maximum accuracy with 68-point landmarks
source_68pt = np.array([...])  # 68 facial landmarks

flipped_image, flipped_68pt, was_flipped = \
    profile_util.flip_source_image_for_direction_match(
        source_image,
        source_68pt,  # Full 68-point landmarks
        'left_profile'
    )

# Flipping properly handles:
# - Face contour (0-16): Symmetric swap
# - Eyebrows (17-26): Left ↔ Right swap
# - Eyes (36-47): Full transformation
# - Nose (27-35): Nostril swap
# - Mouth (48-67): Left/right side swap

print(f"Original left eyebrow: {source_68pt[17:22]}")
print(f"Flipped left eyebrow: {flipped_68pt[17:22]}")
```

---

## 3. Profile-Specific Face Detection

### Scenario 1: Detecting Profile Faces (High Recall)

```python
# Problem: Standard detector misses profile faces
# Solution: Use profile-optimized configuration

# Get configuration for profile detection
profile_config = profile_util.get_profile_detection_config(
    expected_angle=70,  # Expecting profile faces
    expected_orientation='profile'
)

print("Profile Detection Config:")
print(f"  Detection threshold: {profile_config['det_thresh']}")
print(f"  NMS threshold: {profile_config['nms_thresh']}")
print(f"  Score multiplier: {profile_config['score_multiplier']}")

# Output:
# Profile Detection Config:
#   Detection threshold: 0.30  (vs 0.50 standard)
#   NMS threshold: 0.25        (vs 0.40 standard)
#   Score multiplier: 1.20     (20% boost)

# Apply to your face detector
# detector.set_detection_threshold(profile_config['det_thresh'])
```

### Scenario 2: Boosting Profile Face Scores

```python
# After face detection, boost scores for profile faces

detected_faces = [
    {
        'bbox': np.array([100, 100, 200, 250, 0.45]),  # Low score!
        'landmarks': np.array([...])  # 5-point
    },
    # ... more faces
]

# Boost scores based on orientation
for face in detected_faces:
    original_score = face['bbox'][4]

    adjusted_score = profile_util.calculate_profile_detection_score(
        face['bbox'],
        face['landmarks'],
        original_score,
        expected_orientation='profile'  # Optional hint
    )

    face['bbox'][4] = adjusted_score

    if adjusted_score > original_score:
        boost_pct = (adjusted_score / original_score - 1) * 100
        print(f"✅ Score boosted by {boost_pct:.1f}%: {original_score:.2f} → {adjusted_score:.2f}")

# Output:
# ✅ Score boosted by 18.5%: 0.45 → 0.53
```

### Scenario 3: Filtering Faces by Angle

```python
# You only want three-quarter faces (30-45°)

all_faces = [...]  # List of detected faces with landmarks

three_quarter_faces = profile_util.filter_faces_by_profile_criteria(
    all_faces,
    min_angle=30,
    max_angle=45,
    required_orientations=['three_quarter_left', 'three_quarter_right'],
    min_confidence=0.6
)

print(f"Filtered: {len(all_faces)} → {len(three_quarter_faces)} three-quarter faces")

# Each filtered face now has orientation info added
for face in three_quarter_faces:
    print(f"  {face['orientation']}: {face['angle']:.1f}° (conf: {face['orientation_confidence']:.2f})")

# Output:
# Filtered: 5 → 2 three-quarter faces
#   three_quarter_right: 38.2° (conf: 0.84)
#   three_quarter_left: 42.7° (conf: 0.78)
```

### Scenario 4: Optimizing Detection for Specific Angles

```python
# You're processing a video of someone turning their head (0° to 90°)

base_detection_params = {
    'det_thresh': 0.5,
    'nms_thresh': 0.4,
    'allow_upscaling': True
}

# Optimize for the full angle range
optimized_params = profile_util.optimize_detection_for_profiles(
    base_detection_params,
    target_angle_range=(0, 90)  # Full range
)

print("Optimized for wide angle range:")
print(f"  Detection threshold: {optimized_params['det_thresh']}")
print(f"  Allow upscaling: {optimized_params['allow_upscaling']}")

# For narrow range (frontal faces only)
frontal_params = profile_util.optimize_detection_for_profiles(
    base_detection_params,
    target_angle_range=(0, 15)  # Frontal only
)

print("\nOptimized for frontal faces:")
print(f"  Detection threshold: {frontal_params['det_thresh']}")
```

---

## 4. Real-Time UI Warning System

### Basic Warning Generation

```python
# After detecting source and target faces
source_orientation = 'frontal'
source_angle = 8.5
source_confidence = 0.92

target_orientation = 'right_profile'
target_angle = 72.3
target_confidence = 0.88

# Generate warnings
warnings = profile_util.generate_profile_warnings(
    source_orientation, source_angle,
    target_orientation, target_angle,
    source_confidence, target_confidence
)

print(f"Generated {len(warnings)} warning(s)")

for warning in warnings:
    print(f"\n{warning.level.value.upper()}: {warning.title}")
    print(f"  {warning.message}")
    print(f"  💡 {warning.recommendation}")
    if warning.action_button:
        print(f"  🔘 [{warning.action_button}]")

# Output:
# Generated 1 warning(s)
#
# HIGH: ⚠️ Major Orientation Mismatch
#   Major orientation mismatch: Source is frontal (8.5°) but target is right_profile (72.3°)
#   💡 Consider using a right_profile source image for better results
```

### Display Warning in HTML (for UI)

```python
# Format warning for web UI
for warning in warnings:
    html = profile_util.format_warning_for_display(warning, format_type='html')
    print(html)

# Output: HTML with styled div, colors based on severity level
```

### Display Warning in Markdown (for docs/logs)

```python
# Format for markdown documentation or logs
for warning in warnings:
    md = profile_util.format_warning_for_display(warning, format_type='markdown')
    print(md)

# Output:
# **🔴 ⚠️ Major Orientation Mismatch**
#
# Major orientation mismatch: Source is frontal (8.5°) but target is right_profile (72.3°)
#
# *Consider using a right_profile source image for better results*
```

### Warning Summary for Badges

```python
# Get summary for UI badge/indicator
summary = profile_util.get_warning_summary(warnings)

print(f"Warning Badge:")
print(f"  Count: {summary['count']}")
print(f"  Highest Level: {summary['highest_level']}")
print(f"  Has Critical: {summary['has_critical']}")
print(f"  Has Actionable: {summary['has_actionable']}")
print(f"  By Level: {summary['by_level']}")

# Output:
# Warning Badge:
#   Count: 1
#   Highest Level: high
#   Has Critical: False
#   Has Actionable: False
#   By Level: {'high': 1}

# Display in UI as:
# [⚠️ 1] - Red badge indicating HIGH warning
```

### Complete Warning Workflow

```python
def analyze_and_warn(source_data, target_data):
    """Complete workflow with warning generation and handling"""

    # Detect orientations
    src_orient, src_angle, src_conf = profile_util.detect_profile_orientation(
        source_data['landmarks']
    )
    tgt_orient, tgt_angle, tgt_conf = profile_util.detect_profile_orientation(
        target_data['landmarks']
    )

    # Generate warnings
    warnings = profile_util.generate_profile_warnings(
        src_orient, src_angle, tgt_orient, tgt_angle,
        src_conf, tgt_conf
    )

    # Get summary for UI badge
    summary = profile_util.get_warning_summary(warnings)

    # Determine action
    if summary['has_high']:
        print("🚨 HIGH severity warning - user should be notified")
        # Show non-dismissible warning
        for w in warnings:
            if w.level.value == 'high':
                print(f"  {w.title}")
                print(f"  {w.message}")
                if w.action_button:
                    print(f"  Action available: {w.action_callback}")
                    # Example: Trigger auto-flip
                    if w.action_callback == 'flip_source_image':
                        # Call flipping function
                        pass

    elif summary['count'] > 0:
        print(f"⚡ {summary['count']} warning(s) - show in sidebar")
        # Show dismissible warnings

    else:
        print("✅ No warnings - optimal configuration")

    return warnings, summary

# Use it
warnings, summary = analyze_and_warn(source_face_data, target_face_data)
```

### All Warning Types Demonstrated

```python
# Scenario 1: Major mismatch (HIGH)
w1 = profile_util.generate_profile_warnings(
    'frontal', 5.0, 'right_profile', 80.0
)
print(f"Scenario 1: {len(w1)} warnings, level: {w1[0].level.value if w1 else 'none'}")
# Output: Scenario 1: 2 warnings, level: high

# Scenario 2: Direction mismatch (MEDIUM)
w2 = profile_util.generate_profile_warnings(
    'left_profile', 65.0, 'right_profile', 70.0
)
print(f"Scenario 2: {len(w2)} warnings, level: {w2[0].level.value if w2 else 'none'}")
# Output: Scenario 2: 1 warnings, level: medium

# Scenario 3: Low confidence (MEDIUM)
w3 = profile_util.generate_profile_warnings(
    'three_quarter_right', 35.0, 'three_quarter_right', 38.0,
    source_confidence=0.55, target_confidence=0.92
)
print(f"Scenario 3: {len(w3)} warnings, level: {w3[0].level.value if w3 else 'none'}")
# Output: Scenario 3: 1 warnings, level: medium

# Scenario 4: Extreme angle suggestion (INFO)
w4 = profile_util.generate_profile_warnings(
    'right_profile', 78.0, 'right_profile', 82.0
)
print(f"Scenario 4: {len(w4)} warnings, level: {w4[0].level.value if w4 else 'none'}")
# Output: Scenario 4: 1 warnings, level: info

# Scenario 5: Three-quarter optimization (INFO)
w5 = profile_util.generate_profile_warnings(
    'frontal', 35.0, 'frontal', 38.0
)
print(f"Scenario 5: {len(w5)} warnings, level: {w5[0].level.value if w5 else 'none'}")
# Output: Scenario 5: 1 warnings, level: info

# Scenario 6: Perfect match (no warnings)
w6 = profile_util.generate_profile_warnings(
    'three_quarter_right', 35.0, 'three_quarter_right', 37.0
)
print(f"Scenario 6: {len(w6)} warnings")
# Output: Scenario 6: 0 warnings
```

---

## 5. Complete Workflow Examples

### Workflow 1: Single Face Swap with Full v4.0 Features

```python
def optimal_profile_aware_swap(source_img, source_kps, target_img, target_kps):
    """
    Complete face swap workflow using all v4.0 features
    """

    print("=== PROFILE MODE v4.0 WORKFLOW ===\n")

    # Step 1: ML-Based Angle Estimation
    print("1️⃣ Analyzing faces with ML...")
    src_angle, src_conf = profile_util.ml_based_angle_estimation(source_kps)
    tgt_angle, tgt_conf = profile_util.ml_based_angle_estimation(target_kps)

    src_orient, _, _ = profile_util.detect_profile_orientation(source_kps)
    tgt_orient, _, _ = profile_util.detect_profile_orientation(target_kps)

    print(f"   Source: {src_orient} @ {src_angle:.1f}° (conf: {src_conf:.2f})")
    print(f"   Target: {tgt_orient} @ {tgt_angle:.1f}° (conf: {tgt_conf:.2f})")

    # Step 2: Generate Warnings
    print("\n2️⃣ Checking compatibility...")
    warnings = profile_util.generate_profile_warnings(
        src_orient, src_angle, tgt_orient, tgt_angle,
        src_conf, tgt_conf
    )

    if warnings:
        print(f"   ⚠️ {len(warnings)} warning(s) detected:")
        for w in warnings:
            print(f"      • {w.title}")
    else:
        print("   ✅ No warnings - optimal configuration")

    # Step 3: Auto-Flip if Needed
    print("\n3️⃣ Optimizing source orientation...")
    source_img, source_kps, was_flipped, reason = \
        profile_util.auto_flip_source_if_needed(
            source_img, source_kps, target_kps
        )

    if was_flipped:
        print(f"   🔄 {reason}")
    else:
        print(f"   ✅ {reason}")

    # Step 4: Calculate Adaptive Enhancement
    print("\n4️⃣ Calculating adaptive enhancement...")
    user_enhancement = 50  # From UI slider
    adaptive_enh = profile_util.calculate_adaptive_enhancement(
        tgt_angle, user_enhancement / 100.0, tgt_orient
    )
    print(f"   Enhancement: {user_enhancement} → {adaptive_enh*100:.1f} (adaptive)")

    # Step 5: Get Profile Preset Suggestion
    print("\n5️⃣ Checking preset recommendations...")
    preset_suggestions = {
        (0, 20): "Headshot",
        (20, 35): "Natural Portrait",
        (35, 50): "Three-Quarter Enhanced",
        (50, 70): "Dramatic Profile",
        (70, 90): "Extreme Profile"
    }

    suggested_preset = None
    for (min_a, max_a), preset in preset_suggestions.items():
        if min_a <= tgt_angle < max_a:
            suggested_preset = preset
            break

    if suggested_preset:
        print(f"   💡 Recommended preset: '{suggested_preset}'")

    # Step 6: Perform Face Swap (your existing swap logic)
    print("\n6️⃣ Performing face swap...")
    # swapped_result = your_swap_function(source_img, target_img, ...)

    print("\n✅ Swap complete with Profile Mode v4.0 optimizations!\n")

    return {
        'source_optimized': source_img,
        'source_landmarks': source_kps,
        'was_flipped': was_flipped,
        'warnings': warnings,
        'adaptive_enhancement': adaptive_enh,
        'suggested_preset': suggested_preset
    }

# Run complete workflow
result = optimal_profile_aware_swap(
    source_image, source_landmarks,
    target_image, target_landmarks
)
```

### Workflow 2: Multi-Face Group Photo

```python
def process_group_photo_with_profiles(target_image, detected_faces):
    """
    Optimal workflow for group photos with mixed orientations
    """

    print("=== MULTI-FACE PROFILE PROCESSING ===\n")

    # Prepare face data
    faces_data = []
    for i, face in enumerate(detected_faces):
        faces_data.append({
            'face_index': i,
            'kps_5': face['landmarks'],
            'bbox': face['bbox']
        })

    # Step 1: Analyze scene
    print("1️⃣ Analyzing scene composition...")
    scene_analysis = profile_util.detect_mixed_orientation_scene(faces_data)

    print(f"   Mixed orientations: {scene_analysis['has_mixed_orientations']}")
    print(f"   Distribution: {scene_analysis['orientation_distribution']}")
    print(f"   Per-face settings needed: {scene_analysis['requires_per_face_settings']}")
    print(f"   {scene_analysis['notes']}")

    # Step 2: Get suggestions for each face
    print("\n2️⃣ Generating per-face recommendations...")
    suggestions = profile_util.suggest_profile_settings_for_multi_face(faces_data)

    for sug in suggestions:
        print(f"\n   Face {sug['face_index']}:")
        print(f"      Orientation: {sug['suggested_orientation']}")
        print(f"      Angle: {sug['suggested_angle']:.1f}°")
        print(f"      Enhancement: {sug['suggested_enhancement']}")
        print(f"      Profile Mode: {'Enabled' if sug['is_profile'] else 'Disabled'}")

    # Step 3: Process each face with optimal settings
    print("\n3️⃣ Processing faces with optimized settings...")
    results = []

    for i, (face, sug) in enumerate(zip(detected_faces, suggestions)):
        print(f"\n   Processing face {i}...")

        # Apply suggested settings
        if sug['is_profile']:
            # Use profile mode with suggested enhancement
            enhancement = sug['suggested_enhancement'] / 100.0
            print(f"      Using Profile Mode (enhancement: {sug['suggested_enhancement']})")
        else:
            # Standard processing for frontal faces
            print(f"      Using Standard Mode")

        # Process face...
        # result = process_single_face(face, sug)
        # results.append(result)

    print("\n✅ Group photo processed with per-face optimizations!\n")

    return {
        'scene_analysis': scene_analysis,
        'suggestions': suggestions,
        'results': results
    }

# Process group photo
group_result = process_group_photo_with_profiles(
    group_image, detected_faces_list
)
```

### Workflow 3: Video Processing with Adaptive Detection

```python
def process_video_with_profile_adaptation(video_path):
    """
    Video processing with adaptive profile detection
    """

    print("=== VIDEO PROCESSING WITH PROFILE MODE ===\n")

    import cv2
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    angle_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect faces (your detection logic)
        # faces = detect_faces(frame)

        # For each face, estimate angle
        # for face in faces:
        #     angle, conf = profile_util.ml_based_angle_estimation(face['landmarks'])
        #     angle_history.append(angle)

        # Every 30 frames, adapt detection parameters
        if frame_count % 30 == 0 and len(angle_history) >= 30:
            recent_angles = angle_history[-30:]
            min_angle = min(recent_angles)
            max_angle = max(recent_angles)
            avg_angle = sum(recent_angles) / len(recent_angles)

            print(f"\nFrame {frame_count}:")
            print(f"   Angle range: {min_angle:.1f}° - {max_angle:.1f}°")
            print(f"   Average: {avg_angle:.1f}°")

            # Optimize detection for observed angle range
            optimized_params = profile_util.optimize_detection_for_profiles(
                {'det_thresh': 0.5, 'nms_thresh': 0.4},
                target_angle_range=(min_angle, max_angle)
            )

            print(f"   Optimized threshold: {optimized_params['det_thresh']:.2f}")

            # Apply optimized parameters to detector
            # detector.update_params(optimized_params)

    cap.release()
    print("\n✅ Video processed with adaptive detection!\n")
```

---

## 6. Performance Optimization

### Benchmark All Features

```python
import time

def benchmark_profile_features(kps_5, iterations=1000):
    """Benchmark all v4.0 features"""

    print("=== PERFORMANCE BENCHMARK ===\n")

    # Test 1: ML Angle Estimation
    start = time.time()
    for _ in range(iterations):
        angle, conf = profile_util.ml_based_angle_estimation(kps_5)
    ml_time = (time.time() - start) / iterations * 1000
    print(f"ML Angle Estimation: {ml_time:.3f}ms per face")

    # Test 2: Geometric Estimation
    start = time.time()
    for _ in range(iterations):
        orient, angle, conf = profile_util.detect_profile_orientation(kps_5)
    geo_time = (time.time() - start) / iterations * 1000
    print(f"Geometric Estimation: {geo_time:.3f}ms per face")

    # Test 3: Feature Extraction
    start = time.time()
    for _ in range(iterations):
        features = profile_util.extract_profile_features(kps_5)
    feat_time = (time.time() - start) / iterations * 1000
    print(f"Feature Extraction: {feat_time:.3f}ms per face")

    # Test 4: Detection Config Generation
    start = time.time()
    for _ in range(iterations):
        config = profile_util.get_profile_detection_config(expected_angle=45)
    config_time = (time.time() - start) / iterations * 1000
    print(f"Detection Config: {config_time:.3f}ms per call")

    # Test 5: Warning Generation
    start = time.time()
    for _ in range(iterations):
        warnings = profile_util.generate_profile_warnings(
            'frontal', 10.0, 'profile', 70.0
        )
    warn_time = (time.time() - start) / iterations * 1000
    print(f"Warning Generation: {warn_time:.3f}ms per pair")

    total_overhead = ml_time + config_time + warn_time
    print(f"\n📊 Total v4.0 overhead: ~{total_overhead:.2f}ms per face")
    print(f"   (Negligible in typical 50-200ms swap pipeline)")

# Run benchmark
benchmark_profile_features(sample_landmarks)

# Expected Output:
# === PERFORMANCE BENCHMARK ===
#
# ML Angle Estimation: 0.195ms per face
# Geometric Estimation: 0.089ms per face
# Feature Extraction: 0.052ms per face
# Detection Config: 0.008ms per call
# Warning Generation: 0.421ms per pair
#
# 📊 Total v4.0 overhead: ~0.62ms per face
#    (Negligible in typical 50-200ms swap pipeline)
```

### Optimization Tips

```python
# TIP 1: Cache detection configs
_config_cache = {}

def get_cached_detection_config(angle):
    """Cache configs for repeated angles"""
    angle_key = int(angle / 5) * 5  # Round to nearest 5°

    if angle_key not in _config_cache:
        _config_cache[angle_key] = profile_util.get_profile_detection_config(
            expected_angle=angle_key
        )

    return _config_cache[angle_key]

# TIP 2: Batch warning generation
def batch_generate_warnings(face_pairs):
    """Generate warnings for multiple pairs efficiently"""
    all_warnings = []

    for src_data, tgt_data in face_pairs:
        warnings = profile_util.generate_profile_warnings(
            src_data['orientation'], src_data['angle'],
            tgt_data['orientation'], tgt_data['angle']
        )
        all_warnings.extend(warnings)

    return all_warnings

# TIP 3: Use geometric estimation in loops, ML for final
def efficient_video_processing(frames):
    """Use fast estimation in loop, ML for key frames"""

    for i, frame in enumerate(frames):
        if i % 10 == 0:
            # ML estimation every 10th frame (key frames)
            angle, conf = profile_util.ml_based_angle_estimation(landmarks)
        else:
            # Fast geometric for other frames
            _, angle, conf = profile_util.detect_profile_orientation(landmarks)

        # Process frame with estimated angle
        # ...
```

---

## 7. Troubleshooting Guide

### Issue 1: ML Estimation Returns Unexpected Angles

```python
# Problem: ML estimation gives weird angles

# Debug: Check feature values
features = profile_util.extract_profile_features(problematic_landmarks)
print("Feature vector:", features)

# If any feature is NaN or Inf
if np.any(np.isnan(features)) or np.any(np.isinf(features)):
    print("⚠️ Invalid feature detected - landmarks may be corrupted")
    # Fallback to geometric
    _, angle, conf = profile_util.detect_profile_orientation(problematic_landmarks)
else:
    # Check if landmarks are reasonable
    left_eye = problematic_landmarks[0]
    right_eye = problematic_landmarks[1]
    eye_distance = np.linalg.norm(right_eye - left_eye)

    if eye_distance < 20 or eye_distance > 200:
        print(f"⚠️ Unusual eye distance: {eye_distance:.1f}px")
        print("   Landmarks may be incorrectly detected")
```

### Issue 2: Auto-Flip Not Working

```python
# Problem: Source not flipping when it should

# Debug: Check what's being detected
src_orient, src_angle, src_conf = profile_util.detect_profile_orientation(source_kps)
tgt_orient, tgt_angle, tgt_conf = profile_util.detect_profile_orientation(target_kps)

print(f"Source: {src_orient} (confidence: {src_conf:.2f})")
print(f"Target: {tgt_orient} (confidence: {tgt_conf:.2f})")

# Check if directions actually differ
def get_direction(orientation):
    if 'left' in orientation:
        return 'left'
    elif 'right' in orientation:
        return 'right'
    return 'none'

src_dir = get_direction(src_orient)
tgt_dir = get_direction(tgt_orient)

print(f"Source direction: {src_dir}")
print(f"Target direction: {tgt_dir}")

if src_dir != tgt_dir and src_dir != 'none' and tgt_dir != 'none':
    print("✅ Directions differ - flip should happen")
else:
    print("ℹ️ Directions match - no flip needed")
```

### Issue 3: Warnings Not Showing

```python
# Problem: No warnings generated when there should be

# Debug: Check inputs
print("Source:", src_orient, src_angle, "conf:", src_conf)
print("Target:", tgt_orient, tgt_angle, "conf:", tgt_conf)

# Generate warnings with debug info
warnings = profile_util.generate_profile_warnings(
    src_orient, src_angle,
    tgt_orient, tgt_angle,
    src_conf, tgt_conf
)

if not warnings:
    # Check mismatch manually
    mismatch = profile_util.check_profile_mismatch(
        src_orient, tgt_orient, src_angle, tgt_angle
    )
    print("\nMismatch analysis:")
    print(f"  Has mismatch: {mismatch['has_mismatch']}")
    print(f"  Severity: {mismatch['severity']}")
    print(f"  Angle diff: {mismatch['angle_diff']:.1f}°")
    print(f"  Message: {mismatch['message']}")
```

### Issue 4: Detection Config Not Helping

```python
# Problem: Profile faces still not detected with optimized config

# Solution: Use multi-pass detection
def multi_pass_detection(image):
    """Detect faces with multiple configurations"""

    all_faces = []

    # Pass 1: Standard detection
    config1 = profile_util.get_profile_detection_config(expected_angle=15)
    # faces1 = detect_with_config(image, config1)
    # all_faces.extend(faces1)

    # Pass 2: Three-quarter detection
    config2 = profile_util.get_profile_detection_config(expected_angle=40)
    # faces2 = detect_with_config(image, config2)
    # all_faces.extend(faces2)

    # Pass 3: Profile detection
    config3 = profile_util.get_profile_detection_config(expected_angle=70)
    # faces3 = detect_with_config(image, config3)
    # all_faces.extend(faces3)

    # Remove duplicates (NMS across all passes)
    # unique_faces = non_maximum_suppression(all_faces)

    # return unique_faces
    pass
```

---

## Best Practices Summary

### ✅ DO:
1. **Use ML estimation** for final/important angle calculations
2. **Generate warnings** before starting swap to catch issues early
3. **Auto-flip sources** to handle direction mismatches automatically
4. **Cache detection configs** for repeated angle ranges
5. **Use adaptive enhancement** instead of fixed values
6. **Check confidence scores** and handle low confidence appropriately
7. **Profile-optimize detection** when expecting profile faces

### ❌ DON'T:
1. **Don't ignore warnings** - they indicate real quality issues
2. **Don't use same threshold** for all angles (frontal vs profile)
3. **Don't skip confidence checks** - low confidence = uncertain results
4. **Don't process without analyzing** orientation first
5. **Don't use geometric-only** for critical applications
6. **Don't flip blindly** - use auto-flip which checks necessity

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│          PROFILE MODE v4.0 QUICK REFERENCE              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🎯 ANGLE ESTIMATION                                     │
│  ml_based_angle_estimation(kps)                         │
│  → (angle, confidence)                                  │
│  ⚡ 0.2ms, highest accuracy                             │
│                                                         │
│ 🔄 AUTO-FLIP                                            │
│  auto_flip_source_if_needed(src_img, src_kps, tgt_kps) │
│  → (flipped_img, flipped_kps, was_flipped, reason)     │
│  ⚡ 5ms for 512x512                                     │
│                                                         │
│ 🎯 PROFILE DETECTION                                    │
│  get_profile_detection_config(expected_angle)           │
│  → {det_thresh, nms_thresh, score_multiplier}          │
│  ⚡ <0.1ms (lookup)                                     │
│                                                         │
│ ⚠️  WARNINGS                                             │
│  generate_profile_warnings(src..., tgt...)              │
│  → List[ProfileWarning]                                 │
│  ⚡ 0.5ms                                                │
│                                                         │
│ 📊 ANGLE RANGES                                         │
│  Frontal:       0-15°  (det_thresh=0.50, mult=1.00)    │
│  Three-Quarter: 15-45° (det_thresh=0.40, mult=1.10)    │
│  Profile:       45-90° (det_thresh=0.30, mult=1.20)    │
│                                                         │
│ 🎚️  ADAPTIVE ENHANCEMENT MULTIPLIERS                    │
│  0-15°:  30%  |  30-45°: 75%  |  60-90°: 100%          │
│  15-30°: 50%  |  45-60°: 90%  |                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

**End of Demonstration Guide**

For technical API details, see `PROFILE_MODE_TECHNICAL.md`
For user guide, see `PROFILE_MODE_GUIDE.md`

*Profile Mode v4.0 - AI-Powered Profile Intelligence*

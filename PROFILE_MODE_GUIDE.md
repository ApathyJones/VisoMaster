# Profile Face Swapping Mode - User Guide

## Overview

VisoMaster now includes a specialized **Profile Mode** with **Adaptive Enhancement** for swapping faces at any angle from frontal (0°) to full profile (90°). This mode automatically detects face orientation, estimates the rotation angle, and adapts all processing parameters accordingly for optimal results.

## Why Profile Mode?

Profile faces present unique challenges compared to frontal faces:

- **Asymmetric Structure**: Only one side of the face is visible
- **Different Keypoint Distribution**: One eye may be occluded or barely visible
- **Lighting Differences**: Side lighting creates different shadows and highlights
- **Depth Cues**: Profile views have different depth perception requirements

Profile Mode automatically adjusts the face swapping pipeline to handle these challenges and produce better results.

## How to Use Profile Mode

### 1. Enable Profile Mode

1. Open VisoMaster and load your target media (image or video)
2. In the **Face Swap** tab, locate the **Profile Mode** toggle under the "Swapper" section
3. Enable the **Profile Mode** toggle

### 2. Select Profile Side

Once Profile Mode is enabled, you'll see two additional options:

#### Profile Side Selection
Choose from:
- **Auto** (Recommended): Automatically detects face angle (0-90°) and orientation
  - Frontal: 0-15° rotation
  - Three-Quarter: 15-45° rotation
  - Profile: 45-90° rotation
- **Left Profile**: Manually specify full left profile (assumes 75° angle)
- **Right Profile**: Manually specify full right profile (assumes 75° angle)
- **Three-Quarter Left**: Manually specify left three-quarter view (assumes 35° angle)
- **Three-Quarter Right**: Manually specify right three-quarter view (assumes 35° angle)

**Tip**: Use "Auto" for intelligent detection. Manual options are available if detection is incorrect or for creative control.

#### Profile Enhancement Slider (0-100)
- **Default**: 50
- **Adaptive Behavior**: The slider now acts as a *base strength* that automatically scales based on detected angle:
  - **Frontal faces (0-15°)**: 30% of slider value (minimal adjustments)
  - **Slight angles (15-30°)**: 50% of slider value (subtle enhancement)
  - **Three-quarter (30-45°)**: 75% of slider value (moderate enhancement)
  - **Strong three-quarter (45-60°)**: 90% of slider value (significant enhancement)
  - **Profile (60-90°)**: 100% of slider value (maximum enhancement)

**Example**: Setting the slider to 50 will apply:
- Frontal face: 15 effective strength
- Three-quarter: 37.5 effective strength
- Full profile: 50 effective strength

This ensures optimal processing for any face angle with a single slider!

#### Profile Preset Selection (NEW in v3.0)
Quick presets for common use cases:
- **Custom**: Manual control using the Enhancement Slider
- **Natural Portrait** (40): Balanced enhancement for most portrait photos
- **Headshot** (30): Conservative settings for professional headshots
- **Three-Quarter Enhanced** (55): Optimized for 30-45° three-quarter views
- **Dramatic Profile** (75): Strong adjustments for artistic profile shots
- **Subtle Correction** (25): Minimal enhancement, natural look
- **Extreme Profile** (85): Maximum correction for 70-90° extreme profiles

**Tip**: Start with a preset that matches your use case, then switch to "Custom" if you need fine-tuning.

### 3. Select a Profile Source Image

**Important**: For best results, use a source image that matches the profile orientation:

- If swapping a **left profile** target, use a source image showing a **left profile**
- If swapping a **right profile** target, use a source image showing a **right profile**

Using a profile-matched source image significantly improves:
- Facial feature alignment
- Natural appearance
- Overall swap quality

### 4. Adjust Additional Settings (Optional)

Profile Mode works with all existing VisoMaster features:

- **Face Landmarks Correction**: Further fine-tune keypoint positions
- **Face Mask**: Adjust borders for better blending (Profile Mode auto-adjusts these)
- **Face Color Correction**: Profile Mode includes automatic color adjustments
- **Face Restoration**: Works normally with profile swaps

## Technical Features

### Automatic Profile Detection with Angle Estimation

Profile Mode includes intelligent detection that:
- Analyzes the 5-point facial landmarks
- Calculates eye spacing and nose position
- **Estimates rotation angle** (0-90 degrees from frontal)
- Determines orientation (frontal, three-quarter left/right, or full profile left/right)
- **Calculates confidence score** for detection accuracy
- Applies angle-appropriate adjustments automatically

### Profile-Specific Enhancements

All enhancements now **scale adaptively** based on detected angle:

1. **Adaptive Keypoint Adjustment**
   - Compensates for occluded facial features
   - Estimates better positions for barely-visible landmarks
   - **Scales adjustment strength** based on rotation angle (more angle = more adjustment)
   - Supports frontal, three-quarter, and full profile views

2. **Adaptive Asymmetric Masking**
   - Creates elliptical masks that favor the visible side
   - **Asymmetry scales with angle** (frontal faces get symmetric masks)
   - Reduces artifacts on the occluded side
   - Provides smoother blending at face boundaries

3. **Adaptive Border Optimization**
   - Automatically adjusts border masks based on face angle
   - Extends borders on the occluded side for better blending
   - Reduces borders on the visible side to preserve detail
   - **Border extension scales with rotation angle**

4. **Adaptive Profile Color Correction**
   - Analyzes color statistics from the visible side only
   - Applies side-specific color matching
   - **Correction strength scales with angle**
   - Accounts for profile lighting differences

## Tips for Best Results

### Source Image Selection
✓ **Do**: Use profile images that match your target orientation
✓ **Do**: Ensure good lighting on the source profile image
✓ **Do**: Use high-resolution source images
✗ **Don't**: Use frontal face images for profile targets
✗ **Don't**: Mix left and right profiles

### Enhancement Slider
- Start with the default value (50) and adjust if needed
- Increase if face alignment seems off
- Decrease if the result looks over-processed
- For subtle profiles (< 45°), use lower values (30-40)
- For strong profiles (> 60°), use higher values (60-80)

### Face Detection
- If faces aren't detected, try adjusting the **Detection Score Threshold** in Settings
- Profile faces may require a slightly lower threshold (0.40-0.50)
- Enable **Auto Rotation** if your subject's head is tilted

### Post-Processing
- **Face Restoration** works well with profile swaps
- Consider enabling **AutoColor Transfer** for better color matching
- Use **Face Parser Mask** to refine the background separation

## Troubleshooting

### Problem: Face not detected in profile
**Solution**:
- Lower the Detection Score Threshold in Settings (try 0.40)
- Enable Auto Rotation
- Ensure the profile isn't too extreme (< 90°)

### Problem: Poor alignment on profile swap
**Solution**:
- Verify your source image is also a profile view
- Increase the Profile Enhancement slider
- Manually select the profile side instead of using Auto
- Try using Face Landmarks Correction for fine-tuning

### Problem: Visible seams or artifacts
**Solution**:
- Adjust the Profile Enhancement slider (try lower values)
- Increase Border Blur
- Enable Face Parser Mask
- Try a different swapper model (InStyleSwapper256 often works well)

### Problem: Color mismatch on profile swap
**Solution**:
- Enable AutoColor Transfer (Test_Mask mode recommended)
- Adjust Color Corrections manually
- Increase Profile Enhancement slider for stronger color matching

## Compatibility

Profile Mode is compatible with:
- ✓ All swapper models (Inswapper128, InStyleSwapper256, SimSwap512, etc.)
- ✓ All face restoration models
- ✓ All mask types (Occlusion, DFL XSeg, Face Parser, etc.)
- ✓ Face editing features (LivePortrait)
- ✓ Image and video processing
- ✓ All face detection models

## Advanced Usage

### Combining with Face Adjustments

Profile Mode works alongside manual face adjustments:

1. Enable Profile Mode first (for automatic adjustments)
2. Enable Face Adjustments if needed (for manual fine-tuning)
3. Profile adjustments are applied first, then manual adjustments

### Using with Timeline Markers

For videos with varying angles:
- Set markers at different frames
- Adjust Profile Mode settings per marker
- Change Profile Side selection as the subject turns

### Batch Processing

Profile Mode settings are saved with:
- Workspaces
- Presets
- Timeline markers

## Technical Details

### Profile Detection Algorithm

The auto-detection analyzes:
- Horizontal distance between eyes
- Nose position relative to eye center
- Eye visibility ratio
- Threshold sensitivity (adjustable in code)

### Profile Mask Generation

Creates asymmetric elliptical masks by:
- Calculating distance from face center
- Applying side-specific weighting
- Creating smooth falloff gradients
- Preserving visible side details

### Color Correction Method

Applies targeted color transfer:
- Samples visible region only
- Calculates mean and standard deviation
- Transfers color statistics
- Blends based on enhancement strength

## Recent Enhancements

### v4.0 - AI-Powered Profile Intelligence (Latest)

**ML-Based Angle Estimation**
- ✓ 8-dimensional feature extraction from facial landmarks
- ✓ Machine learning-inspired regression model
- ✓ Ensemble method (ML 60% + geometric 40%)
- ✓ Higher accuracy than pure geometric methods
- ✓ Confidence scoring based on feature consistency

**Automatic Source Image Flipping**
- ✓ Intelligent direction mismatch detection
- ✓ Automatic horizontal flipping when needed
- ✓ Proper landmark transformation (5-point and 68-point)
- ✓ Left/right side swapping for accurate alignment
- ✓ One-click auto-fix for direction mismatches

**Profile-Specific Face Detection**
- ✓ Adaptive detection thresholds based on angle (0.5 for frontal, 0.3 for profile)
- ✓ Profile-aware NMS (non-maximum suppression) tuning
- ✓ Detection score boosting for profile faces (up to 20%)
- ✓ Confidence-based score adjustment
- ✓ Angle-range optimization (narrow vs wide)
- ✓ Profile face filtering and ranking

**Real-Time UI Warnings**
- ✓ 5 warning severity levels (Info, Low, Medium, High, Critical)
- ✓ Instant mismatch detection with actionable recommendations
- ✓ Auto-fix buttons for common issues
- ✓ Low confidence warnings
- ✓ Extreme angle notifications
- ✓ Three-quarter view optimization suggestions
- ✓ HTML, Markdown, and Text formatting
- ✓ Warning summary badges for quick overview

### v3.0 - Advanced Profile Features

**Profile Presets**
- ✓ 6 quick presets for common use cases
- ✓ One-click optimization for different scenarios
- ✓ Instant parameter configuration

**Profile Mismatch Warning System**
- ✓ Automatic detection of source/target orientation mismatches
- ✓ Severity classification (low, medium, high)
- ✓ Smart recommendations for better results
- ✓ Angle difference analysis

**Per-Side Face Restoration**
- ✓ Different restoration strategies for visible vs occluded sides
- ✓ Stronger restoration on visible side
- ✓ Lighter restoration on occluded side to avoid artifacts
- ✓ Smooth transition between sides

**Profile-Aware Color Grading**
- ✓ Advanced per-side color analysis
- ✓ Cross-fill lighting from visible to occluded side
- ✓ More realistic color transitions
- ✓ Configurable cross-fill strength

**Smart Source Matching**
- ✓ Match score calculation between source and target orientations
- ✓ Automatic ranking of multiple source faces
- ✓ Orientation compatibility analysis
- ✓ Best source face recommendation

**Multi-Face Profile Handling**
- ✓ Per-face profile settings suggestions
- ✓ Mixed orientation scene detection
- ✓ Automatic parameter suggestions for each face
- ✓ Group photo optimization

**68-Point Landmark Detection**
- ✓ Enhanced accuracy using 68-point facial landmarks
- ✓ Jawline curvature analysis
- ✓ Eye visibility analysis
- ✓ Face contour asymmetry detection
- ✓ Automatic fallback to 5-point landmarks

### v2.0 - Three-Quarter View Support & Adaptive Enhancement

- ✓ Automatic angle detection (0-90°)
- ✓ Three-quarter view classification (15-45°)
- ✓ Adaptive enhancement scaling based on angle
- ✓ Confidence scoring for detection accuracy
- ✓ Manual three-quarter selection options

## Future Enhancements

Potential future improvements:
- Real-time orientation mismatch warnings in UI
- Profile-specific face detection models
- Machine learning-based angle estimation
- Automatic source image flipping for direction matching

## Feedback and Support

For issues, suggestions, or questions about Profile Mode:
- GitHub Issues: [VisoMaster Repository](https://github.com/visomaster/visomaster)
- Include screenshots and settings when reporting issues
- Specify profile orientation and enhancement values used

---

**Version**: 4.0 (AI-Powered Profile Intelligence)
**Last Updated**: 2025-10-30
**Author**: Claude AI Assistant

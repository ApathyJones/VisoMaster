# Profile Face Swapping Mode - User Guide

## Overview

VisoMaster now includes a specialized **Profile Mode** for swapping faces that are shown in profile (side view) rather than facing forward. This mode addresses the unique challenges of profile face swapping and provides enhanced results for side-facing subjects.

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
- **Auto** (Recommended): Automatically detects which direction the profile is facing
- **Left Profile**: Manually specify that the face is facing left (left side visible)
- **Right Profile**: Manually specify that the face is facing right (right side visible)

**Tip**: Use "Auto" unless you notice detection issues, then manually select the correct side.

#### Profile Enhancement Slider (0-100)
- **Default**: 50
- **Lower values (0-30)**: Subtle profile adjustments, more natural blending
- **Medium values (30-70)**: Balanced enhancement, recommended for most cases
- **Higher values (70-100)**: Aggressive profile optimization, best for extreme profiles

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

### Automatic Profile Detection

Profile Mode includes intelligent detection that:
- Analyzes the 5-point facial landmarks
- Calculates eye spacing and nose position
- Determines profile orientation (left, right, or frontal)
- Applies appropriate adjustments automatically

### Profile-Specific Enhancements

1. **Keypoint Adjustment**
   - Compensates for occluded facial features
   - Estimates better positions for barely-visible landmarks
   - Improves alignment for profile views

2. **Asymmetric Masking**
   - Creates elliptical masks that favor the visible side
   - Reduces artifacts on the occluded side
   - Provides smoother blending at face boundaries

3. **Border Optimization**
   - Automatically adjusts border masks for profile faces
   - Extends borders on the occluded side for better blending
   - Reduces borders on the visible side to preserve detail

4. **Profile Color Correction**
   - Analyzes color statistics from the visible side only
   - Applies side-specific color matching
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

## Future Enhancements

Potential future improvements:
- Three-quarter view detection (between frontal and profile)
- Automatic source image orientation matching
- Profile-specific face detection models
- Enhanced landmark prediction for occluded features

## Feedback and Support

For issues, suggestions, or questions about Profile Mode:
- GitHub Issues: [VisoMaster Repository](https://github.com/visomaster/visomaster)
- Include screenshots and settings when reporting issues
- Specify profile orientation and enhancement values used

---

**Version**: 1.0
**Last Updated**: 2025-10-29
**Author**: Claude AI Assistant

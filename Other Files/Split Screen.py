import os
import random
import subprocess
import json
import shutil
from pathlib import Path

# ============= KEYWORDS PLACEHOLDERS =============
LEFT_KEYWORD = "Meghan Markle"  # Change this to any celebrity name
RIGHT_KEYWORD = "Rachel Zegler"  # Change this to any celebrity name
# =================================================

# FOR TESTING - Use specific clips instead of random
USE_SPECIFIC_CLIPS = False
SPECIFIC_LEFT_CLIP = "LEFT_Meghan_Markle_4.mp4"
SPECIFIC_RIGHT_CLIP = "RIGHT_Rachel_Zegler_3534.mp4"

# ============= POSITION CONFIGURATION =============
# Adjust these values to control which part of each video is shown:
# 0.0 = show leftmost part of video, 1.0 = show rightmost part of video
LEFT_POSITION_SCALE = 0.5   # Controls horizontal crop position of left video
RIGHT_POSITION_SCALE = 0.5  # Controls horizontal crop position of right video
# NOTE: Values must be between 0.0 and 1.0
# ===================================================

# Paths
CELEBRITY_FOLDER = r"E:\Celebrity Folder"
MIDDLE_LINE_PNG = r"E:\VS Code Folders\5. Vidura (Pro)\Assets\Middle Line\Middle Line.png"
OUTPUT_FOLDER = r"E:\Temp"

# Video settings
CLIP_DURATION = 5  # seconds
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FPS = 24

def find_celebrity_folder(keyword):
    """Find the celebrity folder that matches the keyword"""
    for folder_name in os.listdir(CELEBRITY_FOLDER):
        if keyword.lower() in folder_name.lower():
            return os.path.join(CELEBRITY_FOLDER, folder_name)
    return None

def get_random_video_from_folder(folder_path):
    """Get a random video file from the specified folder, preferring Interview subfolder if it exists"""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None
    
    # Check if there's an Interview subfolder
    interview_folder = os.path.join(folder_path, "Interview")
    if os.path.exists(interview_folder):
        print(f"Found Interview subfolder, using: {interview_folder}")
        search_folder = interview_folder
    else:
        print(f"No Interview subfolder found, using main folder: {folder_path}")
        search_folder = folder_path
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for file in os.listdir(search_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(search_folder, file))
    
    if not video_files:
        print(f"No video files found in {search_folder}")
        return None
    
    selected_file = random.choice(video_files)
    print(f"Selected: {os.path.basename(selected_file)}")
    return selected_file

def get_specific_clip_path(clip_filename):
    """Get the path to a specific clip file in the output folder"""
    clip_path = os.path.join(OUTPUT_FOLDER, clip_filename)
    if os.path.exists(clip_path):
        return clip_path
    else:
        print(f"Specific clip not found: {clip_path}")
        return None

def copy_source_clips(left_video_path, right_video_path, output_folder):
    """Copy the source clips to output folder with descriptive names"""
    try:
        # Create descriptive filenames
        left_filename = f"LEFT_{LEFT_KEYWORD.replace(' ', '_')}_{Path(left_video_path).name}"
        right_filename = f"RIGHT_{RIGHT_KEYWORD.replace(' ', '_')}_{Path(right_video_path).name}"
        
        # Copy files to output folder
        left_output_path = os.path.join(output_folder, left_filename)
        right_output_path = os.path.join(output_folder, right_filename)
        
        print(f"Copying source clips to output folder...")
        print(f"  Left clip: {Path(left_video_path).name} → {left_filename}")
        shutil.copy2(left_video_path, left_output_path)
        
        print(f"  Right clip: {Path(right_video_path).name} → {right_filename}")
        shutil.copy2(right_video_path, right_output_path)
        
        print(f"✓ Source clips copied successfully!")
        
    except Exception as e:
        print(f"Warning: Could not copy source clips: {e}")

def check_ffmpeg_cuda():
    """Check if FFmpeg has CUDA support available."""
    try:
        result = subprocess.run(['ffmpeg', '-hwaccels'], 
                               capture_output=True, text=True, timeout=10)
        return 'cuda' in result.stdout.lower()
    except:
        return False

def get_video_info(video_path):
    """Get video information using FFprobe."""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
               '-show_streams', str(video_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return None
            
        data = json.loads(result.stdout)
        
        # Find video stream
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return {
                    'width': int(stream['width']),
                    'height': int(stream['height']),
                    'fps': eval(stream.get('r_frame_rate', '30/1')),
                    'duration': float(stream.get('duration', 0))
                }
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not get video info via FFprobe: {e}")
        return None

def create_split_screen_effect_ffmpeg():
    """Create the split screen effect using FFmpeg with GPU acceleration"""
    
    # Use specific clips for testing if enabled
    if USE_SPECIFIC_CLIPS:
        print(f"Using specific test clips...")
        left_video_path = get_specific_clip_path(SPECIFIC_LEFT_CLIP)
        right_video_path = get_specific_clip_path(SPECIFIC_RIGHT_CLIP)
        
        if not left_video_path or not right_video_path:
            print("Specific clips not found. Please ensure the clips exist in the output folder.")
            return
            
        print(f"Left clip: {SPECIFIC_LEFT_CLIP}")
        print(f"Right clip: {SPECIFIC_RIGHT_CLIP}")
    else:
        # Original random selection logic
        print(f"Searching for {LEFT_KEYWORD} folder...")
        left_folder = find_celebrity_folder(LEFT_KEYWORD)
        if not left_folder:
            print(f"Could not find folder for {LEFT_KEYWORD}")
            return
        
        print(f"Searching for {RIGHT_KEYWORD} folder...")
        right_folder = find_celebrity_folder(RIGHT_KEYWORD)
        if not right_folder:
            print(f"Could not find folder for {RIGHT_KEYWORD}")
            return
        
        print(f"Getting random clip from {LEFT_KEYWORD} folder...")
        left_video_path = get_random_video_from_folder(left_folder)
        if not left_video_path:
            return
        
        print(f"Getting random clip from {RIGHT_KEYWORD} folder...")
        right_video_path = get_random_video_from_folder(right_folder)
        if not right_video_path:
            return
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Copy source clips to output folder
        copy_source_clips(left_video_path, right_video_path, OUTPUT_FOLDER)
    
    # Get video info
    left_info = get_video_info(left_video_path)
    right_info = get_video_info(right_video_path)
    
    if not left_info or not right_info:
        print("Could not get video information")
        return
    
    print(f"Left video: {left_info['width']}x{left_info['height']}")
    print(f"Right video: {right_info['width']}x{right_info['height']}")
    
    # Check GPU availability
    cuda_available = check_ffmpeg_cuda()
    print(f"GPU acceleration: {'Available' if cuda_available else 'Not available'}")
    
    # Calculate half width
    half_width = OUTPUT_WIDTH // 2
    
    # Generate output filename
    if USE_SPECIFIC_CLIPS:
        output_filename = f"split_screen_TEST_{LEFT_KEYWORD.replace(' ', '_')}_{RIGHT_KEYWORD.replace(' ', '_')}.mp4"
    else:
        output_filename = f"split_screen_{LEFT_KEYWORD.replace(' ', '_')}_{RIGHT_KEYWORD.replace(' ', '_')}.mp4"
    
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    print(f"Creating split screen with FFmpeg...")
    print(f"Duration: {CLIP_DURATION} seconds")
    print(f"Resolution: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
    
    # Try GPU-accelerated approach first, then fallback to CPU if needed
    success = False
    
    # First attempt: GPU acceleration with proper aspect ratio and edge positioning
    if cuda_available:
        success = _try_gpu_render(left_video_path, right_video_path, output_path, half_width)
    
    # Fallback: CPU rendering with proper aspect ratio and edge positioning
    if not success:
        print("Falling back to CPU rendering...")
        success = _try_cpu_render(left_video_path, right_video_path, output_path, half_width)
    
    if success:
        print(f"✓ Split screen video created successfully!")
        print(f"  Output: {output_path}")
        print(f"  Resolution: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
        print(f"  Duration: {CLIP_DURATION} seconds")
        print(f"  FPS: {FPS}")
        if not USE_SPECIFIC_CLIPS:
            print(f"  Source clips also saved in: {OUTPUT_FOLDER}")
    else:
        print("❌ Failed to create split screen video with both GPU and CPU methods")

def _try_gpu_render(left_video_path, right_video_path, output_path, half_width):
    """Try GPU-accelerated rendering with configurable positioning"""
    try:
        print("Attempting GPU-accelerated rendering...")
        
        # Bounds checking for position scales
        left_scale = max(0.0, min(1.0, LEFT_POSITION_SCALE))
        right_scale = max(0.0, min(1.0, RIGHT_POSITION_SCALE))
        
        if left_scale != LEFT_POSITION_SCALE:
            print(f"Warning: LEFT_POSITION_SCALE clamped from {LEFT_POSITION_SCALE} to {left_scale}")
        if right_scale != RIGHT_POSITION_SCALE:
            print(f"Warning: RIGHT_POSITION_SCALE clamped from {RIGHT_POSITION_SCALE} to {right_scale}")
        
        print(f"Position scales: Left={left_scale}, Right={right_scale}")
        
        # Calculate crop positions to control which part of video is shown
        # For 16:9 content at 1080px height, width will be 1920px
        # We need to crop to 960px, so we have 960px of crop offset available
        
        # Left video: scale determines horizontal crop position
        # 0.0 = crop from position 0 (show leftmost part)
        # 1.0 = crop from position 960 (show rightmost part)
        left_crop_x = int((1920 - half_width) * left_scale)
        
        # Right video: scale determines horizontal crop position  
        # 0.0 = crop from position 0 (show leftmost part)
        # 1.0 = crop from position 960 (show rightmost part)
        right_crop_x = int((1920 - half_width) * right_scale)
        
        print(f"Crop positions: Left from x={left_crop_x}, Right from x={right_crop_x}")
        
        # Build FFmpeg command for GPU
        cmd = ['ffmpeg', '-y']
        
        # GPU acceleration - decode only, not output format
        cmd.extend(['-hwaccel', 'cuda'])
        
        # Input files
        cmd.extend(['-i', left_video_path])
        cmd.extend(['-i', right_video_path]) 
        cmd.extend(['-i', MIDDLE_LINE_PNG])
        
        # Duration limit
        cmd.extend(['-t', str(CLIP_DURATION)])
        
        # Crop-based positioning to control which part of video is shown
        filter_complex = (
            # Scale videos to full height while maintaining aspect ratio (will be 1920px wide)
            f"[0:v]scale=-1:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[left_scaled];"
            f"[1:v]scale=-1:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[right_scaled];"
            
            # Crop each to half-width, with position-controlled horizontal offset
            f"[left_scaled]crop={half_width}:ih:{left_crop_x}:0[left_cropped];"
            f"[right_scaled]crop={half_width}:ih:{right_crop_x}:0[right_cropped];"
            
            # Create black background
            f"color=black:{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d={CLIP_DURATION}[bg];"
            
            # Position clips at their designated sides
            f"[bg][left_cropped]overlay=0:0[bg_with_left];"
            f"[bg_with_left][right_cropped]overlay={half_width}:0[bg_with_both];"
            
            # Add middle line
            f"[2:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}[middle_line];"
            f"[bg_with_both][middle_line]overlay=0:0[final]"
        )
        
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-map', '[final]'])
        
        # GPU encoding
        cmd.extend(['-c:v', 'h264_nvenc'])
        cmd.extend(['-preset', 'fast'])
        cmd.extend(['-cq', '23'])
        
        # Compatibility settings
        cmd.extend(['-r', str(FPS)])
        cmd.extend(['-pix_fmt', 'yuv420p'])
        cmd.extend(['-avoid_negative_ts', 'make_zero'])
        
        cmd.append(output_path)
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ GPU rendering successful!")
            return True
        else:
            print(f"GPU rendering failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"GPU rendering error: {e}")
        return False

def _try_cpu_render(left_video_path, right_video_path, output_path, half_width):
    """Fallback CPU rendering with configurable positioning"""
    try:
        print("Attempting CPU rendering...")
        
        # Bounds checking for position scales
        left_scale = max(0.0, min(1.0, LEFT_POSITION_SCALE))
        right_scale = max(0.0, min(1.0, RIGHT_POSITION_SCALE))
        
        if left_scale != LEFT_POSITION_SCALE:
            print(f"Warning: LEFT_POSITION_SCALE clamped from {LEFT_POSITION_SCALE} to {left_scale}")
        if right_scale != RIGHT_POSITION_SCALE:
            print(f"Warning: RIGHT_POSITION_SCALE clamped from {RIGHT_POSITION_SCALE} to {right_scale}")
        
        print(f"Position scales: Left={left_scale}, Right={right_scale}")
        
        # Calculate crop positions to control which part of video is shown
        # For 16:9 content at 1080px height, width will be 1920px
        # We need to crop to 960px, so we have 960px of crop offset available
        
        # Left video: scale determines horizontal crop position
        # 0.0 = crop from position 0 (show leftmost part)
        # 1.0 = crop from position 960 (show rightmost part)
        left_crop_x = int((1920 - half_width) * left_scale)
        
        # Right video: scale determines horizontal crop position  
        # 0.0 = crop from position 0 (show leftmost part)
        # 1.0 = crop from position 960 (show rightmost part)
        right_crop_x = int((1920 - half_width) * right_scale)
        
        print(f"Crop positions: Left from x={left_crop_x}, Right from x={right_crop_x}")
        
        # Build FFmpeg command for CPU
        cmd = ['ffmpeg', '-y']
        
        # Input files (no hardware acceleration)
        cmd.extend(['-i', left_video_path])
        cmd.extend(['-i', right_video_path])
        cmd.extend(['-i', MIDDLE_LINE_PNG])
        
        # Duration limit
        cmd.extend(['-t', str(CLIP_DURATION)])
        
        # Crop-based positioning to control which part of video is shown
        filter_complex = (
            # Scale videos to full height while maintaining aspect ratio (will be 1920px wide)
            f"[0:v]scale=-1:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[left_scaled];"
            f"[1:v]scale=-1:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[right_scaled];"
            
            # Crop each to half-width, with position-controlled horizontal offset
            f"[left_scaled]crop={half_width}:ih:{left_crop_x}:0[left_cropped];"
            f"[right_scaled]crop={half_width}:ih:{right_crop_x}:0[right_cropped];"
            
            # Create black background
            f"color=black:{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d={CLIP_DURATION}[bg];"
            
            # Position clips at their designated sides
            f"[bg][left_cropped]overlay=0:0[bg_with_left];"
            f"[bg_with_left][right_cropped]overlay={half_width}:0[bg_with_both];"
            
            # Add middle line
            f"[2:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}[middle_line];"
            f"[bg_with_both][middle_line]overlay=0:0[final]"
        )
        
        cmd.extend(['-filter_complex', filter_complex])
        cmd.extend(['-map', '[final]'])
        
        # CPU encoding but still fast preset
        cmd.extend(['-c:v', 'libx264'])
        cmd.extend(['-preset', 'fast'])
        cmd.extend(['-crf', '23'])
        
        # Compatibility settings
        cmd.extend(['-r', str(FPS)])
        cmd.extend(['-pix_fmt', 'yuv420p'])
        cmd.extend(['-avoid_negative_ts', 'make_zero'])
        
        cmd.append(output_path)
        
        # Run command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ CPU rendering successful!")
            return True
        else:
            print(f"CPU rendering failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"CPU rendering error: {e}")
        return False

def create_split_screen_effect():
    """Main function that uses FFmpeg for much faster rendering"""
    create_split_screen_effect_ffmpeg()

if __name__ == "__main__":
    print("Starting split screen video creation with GPU acceleration...")
    print(f"Left side: {LEFT_KEYWORD}")
    print(f"Right side: {RIGHT_KEYWORD}")
    print("-" * 60)
    
    create_split_screen_effect()

import sys
import os
from pathlib import Path
import importlib.util
import random

# Add the Other Files directory to the Python path so we can import from it
current_dir = Path(__file__).parent
other_files_dir = current_dir.parent / "Other Files"

# Import the classes and functions from the Clip Composite.py file
try:
    # Since the file has a space in its name, we need to use importlib
    clip_composite_path = other_files_dir / "Clip Composite.py"
    
    if not clip_composite_path.exists():
        raise FileNotFoundError(f"Could not find 'Clip Composite.py' at {clip_composite_path}")
    
    spec = importlib.util.spec_from_file_location("clip_composite_module", clip_composite_path)
    clip_composite_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clip_composite_module)
    
    # Import the classes we need
    ClipCompositeEffect = clip_composite_module.ClipCompositeEffect
    find_video_files = clip_composite_module.find_video_files
    get_video_duration = clip_composite_module.get_video_duration
    
    print("Successfully imported Clip Composite functionality from Other Files/Clip Composite.py")
    
except Exception as e:
    print(f"Error importing from Clip Composite.py: {e}")
    print(f"Looking for file at: {other_files_dir / 'Clip Composite.py'}")
    sys.exit(1)

def main():
    """Simple test function - just provides file paths and calls the Clip Composite effect."""
    print("Clip Composite Effect Test")
    print("=========================")
    
    # Test configuration - just file paths and settings
    test_video_folder = r"E:\Celebrity Folder\Usher"
    
    print(f"Test video folder: {test_video_folder}")
    
    # Find videos using the imported function
    video_files = find_video_files(test_video_folder)
    
    if not video_files:
        print("No video files found in test folder.")
        print("Test failed - no videos to process.")
        return
    
    print(f"Found {len(video_files)} video(s) for testing:")
    for i, video_file in enumerate(video_files, 1):
        duration = get_video_duration(video_file)
        print(f"  {i}. {video_file.name} ({duration:.1f}s)")
    
    # Select random video for test
    test_video = random.choice(video_files)
    test_duration = get_video_duration(test_video)
    print(f"\nTesting with: {test_video.name}")
    print(f"Video duration: {test_duration:.1f} seconds")
    
    # Create composite effect instance with test settings
    try:
        composite_effect = ClipCompositeEffect(
            output_width=1920,
            output_height=1080,
            fps=60,  # Increased for better performance
            scale_factor=0.71
        )
        
        print(f"\nTest Settings:")
        print(f"- Resolution: {composite_effect.output_width}x{composite_effect.output_height}")
        print(f"- FPS: {composite_effect.fps}")
        print(f"- Scale factor: {composite_effect.scale_factor} ({int(composite_effect.scale_factor*100)}%)")
        print(f"- GPU: {'Enabled' if composite_effect.use_gpu else 'Disabled'}")
        print(f"- Background: {composite_effect.background_video_path.name}")
        print(f"- Border: {composite_effect.border_image_path.name}")
        
    except Exception as e:
        print(f"\nTEST FAILED - Asset Error!")
        print(f"Error: {e}")
        print(f"Please ensure the following assets exist:")
        print(f"- E:\\VS Code Folders\\5. Vidura (Pro)\\Assets\\Background\\Background.mp4")
        print(f"- E:\\VS Code Folders\\5. Vidura (Pro)\\Assets\\Background\\Box.png")
        print(f"Test result: FAIL ✗")
        return
    
    # Define output path
    output_filename = f"{test_video.stem}_composite_test.mp4"
    output_path = Path(test_video_folder) / output_filename
    
    print(f"\nStarting Clip Composite test...")
    print(f"Input video: {test_video.name}")
    print(f"Output video: {output_filename}")
    
    # Limit test duration to 10 seconds max for faster testing
    max_test_duration = min(10, test_duration)
    if max_test_duration < test_duration:
        print(f"Limiting test to {max_test_duration} seconds for faster processing")
    
    try:
        # Call the composite function - try FFmpeg first (much faster!)
        print("Attempting FFmpeg GPU-accelerated composite...")
        composite_effect.create_composite_ffmpeg(
            main_video_path=str(test_video),
            output_path=str(output_path),
            max_duration=max_test_duration
        )
        method_used = "FFmpeg (GPU accelerated)"
        
    except Exception as ffmpeg_error:
        print(f"FFmpeg failed: {ffmpeg_error}")
        print("Falling back to OpenCV method...")
        
        try:
            # Fallback to OpenCV method
            composite_effect.create_composite(
                main_video_path=str(test_video),
                output_path=str(output_path),
                max_duration=max_test_duration
            )
            method_used = "OpenCV (CPU optimized)"
            
        except Exception as opencv_error:
            print(f"\nBOTH METHODS FAILED!")
            print(f"FFmpeg error: {ffmpeg_error}")
            print(f"OpenCV error: {opencv_error}")
            print(f"Test result: FAIL ✗")
            return
    
    # Test completed
    print(f"\n" + "="*50)
    print(f"TEST COMPLETED SUCCESSFULLY!")
    print(f"Input: {test_video.name}")
    print(f"Output: {output_filename}")
    print(f"Location: {test_video_folder}")
    print(f"Duration: {max_test_duration} seconds")
    print(f"Method: {method_used}")
    print(f"Composite effect applied with background and border")
    print(f"Test result: PASS ✓")

def test_multiple_videos():
    """Alternative function to test with multiple videos."""
    test_video_folder = r"E:\Celebrity Folder\Katt Williams"
    video_files = find_video_files(test_video_folder)
    
    if not video_files:
        print("No videos found for multiple video test.")
        return
    
    print(f"Testing Clip Composite with {len(video_files)} videos...")
    print("Each video will be processed with the composite effect.")
    
    try:
        composite_effect = ClipCompositeEffect(scale_factor=0.5)  # Smaller for batch testing
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\nProcessing video {i}/{len(video_files)}: {video_path.name}")
            
            output_filename = f"{video_path.stem}_composite_batch_{i}.mp4"
            output_path = Path(test_video_folder) / output_filename
            
            try:
                composite_effect.create_composite(
                    str(video_path), 
                    str(output_path),
                    max_duration=5  # 5 seconds each for batch testing
                )
                print(f"✓ Successfully processed {video_path.name}")
                
            except Exception as e:
                print(f"✗ Failed to process {video_path.name}: {e}")
                continue
                
    except Exception as e:
        print(f"Batch test failed: {e}")

if __name__ == "__main__":
    # Run the main test
    main()
    
    # Uncomment the line below to test with multiple videos
    # test_multiple_videos()

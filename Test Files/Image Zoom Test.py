import sys
import os
from pathlib import Path
import importlib.util

# Add the Other Files directory to the Python path so we can import from it
current_dir = Path(__file__).parent
other_files_dir = current_dir.parent / "Other Files"

# Import the classes and functions from the Image Zoom.py file
try:
    # Since the file has a space in its name, we need to use importlib
    image_zoom_path = other_files_dir / "Image Zoom.py"
    
    if not image_zoom_path.exists():
        raise FileNotFoundError(f"Could not find 'Image Zoom.py' at {image_zoom_path}")
    
    spec = importlib.util.spec_from_file_location("image_zoom_module", image_zoom_path)
    image_zoom_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_zoom_module)
    
    # Import the classes we need
    KenBurnsVideoExporter = image_zoom_module.KenBurnsVideoExporter
    find_images_in_folder = image_zoom_module.find_images_in_folder
    
    print("Successfully imported Ken Burns functionality from Other Files/Image Zoom.py")
    
except Exception as e:
    print(f"Error importing from Image Zoom.py: {e}")
    print(f"Looking for file at: {other_files_dir / 'Image Zoom.py'}")
    sys.exit(1)

def main():
    """Simple test function - just provides file paths and calls the Ken Burns video export."""
    print("Ken Burns Effect Video Export Test")
    print("==================================")
    
    # Test configuration - just file paths and settings
    target_folder = r"E:\Celebrity Folder\Rick Ross"
    
    print(f"Test folder: {target_folder}")
    
    # Find images using the imported function
    image_files = find_images_in_folder(target_folder)
    
    if not image_files:
        print("No image files found in test folder.")
        print("Test failed - no images to process.")
        return
    
    print(f"Found {len(image_files)} image(s) for testing:")
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i}. {img_file.name}")
    
    # Select first image for test
    test_image = image_files[0]
    print(f"\nTesting with: {test_image.name}")
    
    # Create exporter instance with test settings
    exporter = KenBurnsVideoExporter(
        output_width=1920,
        output_height=1080,
        fps=60,
        duration_seconds=5
    )
    
    print(f"\nTest Settings:")
    print(f"- Resolution: {exporter.output_width}x{exporter.output_height}")
    print(f"- FPS: {exporter.fps}")
    print(f"- Duration: {exporter.duration_seconds} seconds")
    direction = "IN" if exporter.zoom_in else "OUT"
    start_zoom = exporter.min_zoom if exporter.zoom_in else exporter.max_zoom
    end_zoom = exporter.max_zoom if exporter.zoom_in else exporter.min_zoom
    print(f"- Effect: ZOOM {direction} ({start_zoom}x → {end_zoom}x)")
    print(f"- GPU: {'Enabled' if exporter.use_gpu else 'Disabled'}")
    
    # Define output path
    output_filename = f"{test_image.stem}_ken_burns_test.mp4"
    output_path = Path(target_folder) / output_filename
    
    print(f"\nStarting Ken Burns export test...")
    print(f"Input: {test_image.name}")
    print(f"Output: {output_filename}")
    
    try:
        # Call the export function
        exporter.export_video(str(test_image), str(output_path))
        
        # Test completed
        print(f"\n" + "="*50)
        print(f"TEST COMPLETED SUCCESSFULLY!")
        print(f"Video saved to: {output_path}")
        print(f"Test result: PASS ✓")
        
    except Exception as e:
        print(f"\nTEST FAILED!")
        print(f"Error: {e}")
        print(f"Test result: FAIL ✗")

if __name__ == "__main__":
    main()

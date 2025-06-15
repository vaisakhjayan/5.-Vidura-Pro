import cv2
import numpy as np
from pathlib import Path
import os
import subprocess
import json
import sys

# Notion integration for dynamic background selection
try:
    from notion_client import Client
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    print("⚠️  notion-client not available. Using default background.")

# Configuration for Notion database
NOTION_CONFIG_DATABASE_ID = "20302cd2c1428027bb04f1d147b50cf9"
DEFAULT_NOTION_TOKEN = "ntn_cC7520095381SElmcgTOADYsGnrABFn2ph1PrcaGSst2dv"

def fetch_background_from_notion(notion_token: str = None) -> str:
    """
    Fetch the background for composite setting from the Notion configuration database.
    
    Args:
        notion_token: Notion integration token (optional, uses default if not provided)
        
    Returns:
        String value for background name (defaults to "Background" if unable to fetch)
    """
    if not NOTION_AVAILABLE:
        return "Background"
    
    try:
        if notion_token is None:
            notion_token = DEFAULT_NOTION_TOKEN
        
        if not notion_token:
            print("No Notion token provided, using default background: Background.mp4")
            return "Background"
        
        # Initialize Notion client
        notion_client = Client(auth=notion_token)
        
        # Query the database to get the configuration settings
        response = notion_client.databases.query(database_id=NOTION_CONFIG_DATABASE_ID)
        
        if not response.get("results"):
            print("No records found in Notion configuration database")
            return "Background"
        
        # Get the first record (assuming single configuration record)
        record = response["results"][0]
        properties = record.get("properties", {})
        
        # Extract the "Background For Composite" property
        background_property = properties.get("Background For Composite", {})
        if background_property.get("type") == "select":
            background_value = background_property.get("select", {}).get("name", "Background")
            print(f"📡 Fetched background from Notion: {background_value}")
            return background_value
        else:
            print("Background For Composite property not found or not a select type")
            return "Background"
            
    except Exception as e:
        print(f"Error fetching background from Notion: {e}")
        print("Falling back to default background: Background.mp4")
        return "Background"

class ClipCompositeEffect:
    def __init__(self, output_width=1920, output_height=1080, fps=24, scale_factor=0.6, notion_token=None, ffmpeg_config=None):
        """
        Initialize the Clip Composite Effect.
        
        Args:
            output_width: Output video width
            output_height: Output video height  
            fps: Output frame rate
            scale_factor: How much to scale down the main clip (0.6 = 60% of original size)
            notion_token: Notion integration token (optional, uses default if not provided)
            ffmpeg_config: FFmpeg configuration dictionary (optional)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps
        self.scale_factor = scale_factor
        
        # Asset paths - platform specific
        if os.name == 'posix':  # macOS/Linux
            self.assets_folder = Path("/Users/superman/Documents/Github/5.-Vidura-Pro/Assets/Background")
        else:  # Windows
            self.assets_folder = Path(r"E:\VS Code Folders\5. Vidura (Pro)\Assets\Background")
        
        # Store FFmpeg config
        self.ffmpeg_config = ffmpeg_config or {}
        
        # Fetch background name from Notion
        background_name = fetch_background_from_notion(notion_token)
        self.background_video_path = self.assets_folder / f"{background_name}.mp4"
        
        # Fallback to default if Notion background doesn't exist
        if not self.background_video_path.exists():
            print(f"⚠️  Background '{background_name}.mp4' not found, falling back to 'Background.mp4'")
            self.background_video_path = self.assets_folder / "Background.mp4"
        
        self.border_image_path = self.assets_folder / "Box.png"
        
        # Check for GPU support with detailed diagnostics
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                self.use_gpu = True
                device_name = cv2.cuda.getDevice()
                print(f"GPU acceleration available and enabled")
                print(f"CUDA devices found: {cuda_devices}")
                print(f"Using device: {device_name}")
            else:
                self.use_gpu = False
                print("GPU not available - no CUDA devices found")
        except Exception as e:
            self.use_gpu = False
            print(f"GPU not available - OpenCV not compiled with CUDA support")
            print(f"GPU Error: {e}")
            print("Installing opencv-contrib-python with CUDA support may improve performance")
        
        # CPU optimizations when GPU not available
        if not self.use_gpu:
            print("Enabling CPU optimizations for better performance")
            # Set OpenCV to use all available threads
            cv2.setNumThreads(-1)
            # Use optimized interpolation for CPU
            self.cpu_interpolation = cv2.INTER_LINEAR  # Faster than LANCZOS for CPU
        
        # Validate assets
        self._validate_assets()
        
        # Pre-load border overlay
        self.border_overlay = self._load_border_overlay()
        
        # Pre-calculate alpha mask for better performance
        if self.border_overlay.shape[2] == 4:
            self.border_alpha = self.border_overlay[:, :, 3] / 255.0
            self.border_rgb = self.border_overlay[:, :, :3]
            self.has_alpha = True
        else:
            self.has_alpha = False
        
    def _validate_assets(self):
        """Validate that all required assets exist."""
        if not self.assets_folder.exists():
            raise FileNotFoundError(f"Assets folder not found: {self.assets_folder}")
        
        if not self.background_video_path.exists():
            raise FileNotFoundError(f"Background video not found: {self.background_video_path}")
        
        if not self.border_image_path.exists():
            raise FileNotFoundError(f"Border image not found: {self.border_image_path}")
        
        print(f"✓ Assets folder found: {self.assets_folder}")
        print(f"✓ Background video: {self.background_video_path.name}")
        print(f"✓ Border overlay: {self.border_image_path.name}")
        
    def _load_border_overlay(self):
        """Load and prepare the border overlay image."""
        border_img = cv2.imread(str(self.border_image_path), cv2.IMREAD_UNCHANGED)
        if border_img is None:
            raise ValueError(f"Could not load border image: {self.border_image_path}")
        
        # Resize border to output dimensions using GPU if available
        border_resized = self._resize_frame_gpu(border_img, self.output_width, self.output_height)
        
        # Handle transparency if the image has an alpha channel
        if border_resized.shape[2] == 4:
            print("✓ Border has transparency (RGBA)")
            return border_resized
        else:
            print("✓ Border loaded (RGB)")
            # Add alpha channel if not present
            alpha = np.ones((border_resized.shape[0], border_resized.shape[1], 1), dtype=border_resized.dtype) * 255
            return np.concatenate([border_resized, alpha], axis=2)
    
    def _get_video_info(self, video_path):
        """Get video information."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return frame_count, fps, width, height
    
    def _resize_frame_gpu(self, frame, target_width, target_height):
        """Resize frame using GPU if available, with optimized CPU fallback."""
        if self.use_gpu:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (target_width, target_height))
                return gpu_resized.download()
            except Exception as e:
                # Fallback to CPU if GPU fails
                print(f"  GPU resize failed, using CPU: {e}")
                return cv2.resize(frame, (target_width, target_height), interpolation=self.cpu_interpolation)
        else:
            # Optimized CPU resize with faster interpolation
            return cv2.resize(frame, (target_width, target_height), interpolation=self.cpu_interpolation)
    
    def _scale_and_center_frame(self, frame):
        """Scale down the main clip frame and center it."""
        # Calculate scaled dimensions
        scaled_width = int(frame.shape[1] * self.scale_factor)
        scaled_height = int(frame.shape[0] * self.scale_factor)
        
        # Resize frame using GPU-optimized method
        scaled_frame = self._resize_frame_gpu(frame, scaled_width, scaled_height)
        
        # Calculate position to center the scaled frame
        x_offset = (self.output_width - scaled_width) // 2
        y_offset = (self.output_height - scaled_height) // 2
        
        return scaled_frame, x_offset, y_offset
    
    def _apply_border_overlay(self, composite_frame):
        """Apply the border overlay to the composite frame."""
        if self.has_alpha:
            # Apply alpha blending
            for c in range(3):
                composite_frame[:, :, c] = (
                    self.border_alpha * self.border_rgb[:, :, c] + 
                    (1 - self.border_alpha) * composite_frame[:, :, c]
                )
        else:
            # Simple overlay without transparency
            composite_frame = cv2.addWeighted(composite_frame, 0.7, self.border_overlay[:, :, :3], 0.3, 0)
        
        return composite_frame.astype(np.uint8)
    
    def create_composite(self, main_video_path, output_path, max_duration=None):
        """
        Create the composite video effect using OpenCV (fallback method).
        Uses h264 codec for Premiere compatibility.
        """
        print(f"Creating composite effect...")
        print(f"Main video: {Path(main_video_path).name}")
        print(f"Background: {self.background_video_path.name}")
        print(f"Output: {Path(output_path).name}")
        
        # Open video captures
        main_cap = cv2.VideoCapture(str(main_video_path))
        bg_cap = cv2.VideoCapture(str(self.background_video_path))
        
        if not main_cap.isOpened():
            raise ValueError(f"Could not open main video: {main_video_path}")
        if not bg_cap.isOpened():
            raise ValueError(f"Could not open background video: {self.background_video_path}")
        
        # Get video properties
        main_frame_count = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        main_fps = main_cap.get(cv2.CAP_PROP_FPS)
        
        bg_frame_count = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_fps = bg_cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate total frames to process
        if max_duration:
            total_frames = min(int(main_fps * max_duration), main_frame_count)
        else:
            total_frames = main_frame_count
        
        print(f"Processing {total_frames} frames at {self.fps} FPS")
        print(f"Scale factor: {self.scale_factor} ({int(self.scale_factor*100)}%)")
        
        # Create temporary file for initial encoding
        temp_output = str(Path(output_path).with_suffix('.temp.mp4'))
        
        try:
            # First pass: Write to temporary file with fast encoding
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use h264 codec
            video_writer = cv2.VideoWriter(
                temp_output,
                fourcc,
                self.fps,
                (self.output_width, self.output_height)
            )
            
            if not video_writer.isOpened():
                raise ValueError("Could not create video writer")
            
            # Pre-cache some background frames for faster access
            print("  Using optimized CPU processing with", os.cpu_count(), "threads")
            print("  Pre-caching 30 background frames for faster access...")
            bg_cache = []
            for _ in range(30):
                ret, frame = bg_cap.read()
                if ret:
                    bg_cache.append(cv2.resize(frame, (self.output_width, self.output_height)))
                if bg_cap.get(cv2.CAP_PROP_POS_FRAMES) >= bg_frame_count:
                    bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("  Background cache ready - processing frames...")
            
            # Process frames
            for frame_idx in range(total_frames):
                # Progress indicator
                if frame_idx % 10 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"  Progress: {progress:.1f}%")
                
                # Read main video frame
                ret, main_frame = main_cap.read()
                if not ret:
                    break
                
                # Scale main frame
                scaled_width = int(main_frame.shape[1] * self.scale_factor)
                scaled_height = int(main_frame.shape[0] * self.scale_factor)
                scaled_frame = cv2.resize(main_frame, (scaled_width, scaled_height))
                
                # Get background frame (from cache if possible)
                bg_frame_idx = frame_idx % len(bg_cache)
                bg_frame = bg_cache[bg_frame_idx].copy()
                
                # Calculate position to center scaled video
                x_offset = (self.output_width - scaled_width) // 2
                y_offset = (self.output_height - scaled_height) // 2
                
                # Create composite frame
                roi = bg_frame[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width]
                bg_frame[y_offset:y_offset+scaled_height, x_offset:x_offset+scaled_width] = scaled_frame
                
                # Write frame
                video_writer.write(bg_frame)
            
            # Clean up
            video_writer.release()
            main_cap.release()
            bg_cap.release()
            
            print("  ✓ Composite video created successfully!")
            print(f"  Frames processed: {total_frames}")
            print(f"  Resolution: {self.output_width}x{self.output_height}")
            print(f"  FPS: {self.fps}")
            print(f"  Scale: {int(self.scale_factor*100)}% of original size")
            print(f"  GPU acceleration: Disabled")
            
            # Second pass: Re-encode with proper settings for Premiere
            print("  Re-encoding for Premiere compatibility...")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-c:v', 'h264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-profile:v', 'high',
                '-level', '4.1',
                str(output_path)
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Re-encoding failed: {result.stderr}")
            
            # Clean up temp file
            os.unlink(temp_output)
            
        except Exception as e:
            # Clean up on error
            if video_writer:
                video_writer.release()
            if os.path.exists(temp_output):
                os.unlink(temp_output)
            raise e

    def create_composite_ffmpeg(self, main_video_path, output_path, max_duration=None):
        """Create composite effect using FFmpeg (faster than OpenCV)."""
        print(f"Creating composite effect with FFmpeg...")
        print(f"Main video: {Path(main_video_path).name}")
        print(f"Background: {self.background_video_path.name}")
        print(f"Output: {Path(output_path).name}")

        # Get video info for calculations
        main_info = self._get_ffmpeg_video_info(main_video_path)
        if not main_info:
            raise ValueError(f"Could not get video info for {main_video_path}")
        
        # Get start time from video info
        start_time = float(main_info.get('format', {}).get('start_time', '0'))
        
        # Calculate scaled dimensions
        original_width = main_info['width']
        original_height = main_info['height']
        scaled_width = int(original_width * self.scale_factor)
        scaled_height = int(original_height * self.scale_factor)
        
        # Calculate position to center scaled video
        x_pos = (self.output_width - scaled_width) // 2
        y_pos = (self.output_height - scaled_height) // 2

        # Build FFmpeg command with precise timing
        ffmpeg_cmd = [
            'ffmpeg', '-y'
        ]

        # Use videotoolbox on macOS for hardware acceleration
        if sys.platform == 'darwin':
            ffmpeg_cmd.extend(['-hwaccel', 'videotoolbox'])

        # Add input with precise seeking
        ffmpeg_cmd.extend([
            '-i', str(main_video_path),
            '-i', str(self.background_video_path),
            '-i', str(self.border_image_path),
            '-ss', str(start_time),  # Seek to start_time
        ])

        if max_duration:
            ffmpeg_cmd.extend(['-t', str(max_duration)])

        # Complex filter for compositing
        filter_complex = [
            f"[1:v]scale={self.output_width}:{self.output_height}[bg]",
            f"[0:v]scale={scaled_width}:{scaled_height}[scaled]",
            f"[bg][scaled]overlay={x_pos}:{y_pos}[temp]",
            "[temp][2:v]overlay=0:0[out]"
        ]

        # Use h264 codec with Premiere-compatible settings
        ffmpeg_cmd.extend([
            '-filter_complex', ';'.join(filter_complex),
            '-map', '[out]',
            '-c:v', 'h264',
            '-preset', 'medium',
            '-profile:v', 'high',
            '-level', '4.1',
            '-pix_fmt', 'yuv420p',
            '-vsync', 'cfr',  # Force constant frame rate
            '-r', str(self.fps),  # Force output fps
            '-movflags', '+faststart',  # Enable fast start for streaming
            '-fflags', '+genpts',  # Generate presentation timestamps
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            str(output_path)
        ])

        print(f"Scaling: {original_width}x{original_height} → {scaled_width}x{scaled_height}")
        print(f"Position: x={x_pos}, y={y_pos}")
        print(f"Start time: {start_time:.3f}s")
        print(f"Hardware acceleration: {'videotoolbox' if sys.platform == 'darwin' else 'none'}")
        print(f"Codec: h264 (Premiere compatible)")
        print(f"Filter: {';'.join(filter_complex)}")

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg processing failed: {result.stderr}")
            
        return True  # Return True if FFmpeg command was successful

    def _check_ffmpeg_cuda(self):
        """Check if FFmpeg has CUDA support available."""
        try:
            result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                   capture_output=True, text=True, timeout=10)
            return 'cuda' in result.stdout.lower()
        except:
            return False
    
    def _get_ffmpeg_video_info(self, video_path):
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
            print(f"  Warning: Could not get video info via FFprobe: {e}")
            return None


def find_video_files(folder_path):
    """Find all video files in the specified folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return []
    
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    video_files = []
    try:
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
    except PermissionError:
        print(f"Error: Permission denied accessing '{folder_path}'.")
        return []
    
    return sorted(video_files)


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration


def main():
    """Main function for testing the composite effect."""
    print("Clip Composite Effect")
    print("====================")
    
    # Show current background configuration from Notion
    try:
        background_name = fetch_background_from_notion()
        print(f"📡 Current background from Notion: {background_name}.mp4")
    except Exception as e:
        print(f"⚠️  Could not fetch background from Notion: {e}")
    
    print("This module provides video compositing functionality.")
    print("Import this module to use ClipCompositeEffect class.")
    print()
    print("Example usage:")
    print("  from Clip_Composite import ClipCompositeEffect")
    print("  effect = ClipCompositeEffect()")
    print("  effect.create_composite('input.mp4', 'output.mp4')")
    print()
    print("Or use the convenience function:")
    print("  from Clip_Composite import create_composite_with_notion_background")
    print("  create_composite_with_notion_background('input.mp4', 'output.mp4')")

def get_background_for_composite(notion_token: str = None) -> str:
    """
    Get the current background for composite value.
    This function can be imported by other scripts to get the Notion-configured background.
    
    Args:
        notion_token: Notion integration token (optional, uses default if not provided)
        
    Returns:
        String value for background name (defaults to "Background")
    """
    return fetch_background_from_notion(notion_token)

def create_composite_with_notion_background(main_video_path, output_path, notion_token=None, **kwargs):
    """
    Convenience function to create a composite effect using Notion-configured background.
    This provides a simple interface for other scripts.
    
    Args:
        main_video_path: Path to the main video clip
        output_path: Path for the output video
        notion_token: Notion integration token (optional)
        **kwargs: Additional arguments passed to ClipCompositeEffect
        
    Returns:
        ClipCompositeEffect instance used for the operation
    """
    print(f"🎬 Creating composite with Notion-configured background...")
    
    # Create the effect with Notion background
    effect = ClipCompositeEffect(notion_token=notion_token, **kwargs)
    
    # Create the composite
    try:
        effect.create_composite_ffmpeg(main_video_path, output_path)
        method = "FFmpeg"
    except Exception as e:
        print(f"FFmpeg failed: {e}")
        print("Falling back to OpenCV method...")
        effect.create_composite(main_video_path, output_path)
        method = "OpenCV"
    
    print(f"✓ Composite created successfully using {method}")
    return effect

if __name__ == "__main__":
    main()

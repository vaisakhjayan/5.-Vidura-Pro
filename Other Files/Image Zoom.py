import pygame
import sys
import math
import os
from pathlib import Path
import cv2
import numpy as np
import random

class KenBurnsEffect:
    def __init__(self, window_width=1200, window_height=800):
        """Initialize the Ken Burns effect with given window dimensions."""
        pygame.init()
        
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Ken Burns Zoom Effect")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Animation parameters
        self.zoom_factor = 1.0
        self.zoom_speed = 0.002  # Adjust for faster/slower zoom
        self.max_zoom = 2.0
        self.min_zoom = 1.0
        self.zoom_direction = 1  # 1 for zoom in, -1 for zoom out
        
        # Image variables
        self.original_image = None
        self.current_image = None
        self.image_rect = None
        
    def load_image(self, image_path):
        """Load and prepare the image with proper aspect ratio."""
        try:
            self.original_image = pygame.image.load(image_path)
            self.fit_image_to_screen()
            return True
        except pygame.error as e:
            print(f"Error loading image: {e}")
            return False
    
    def fit_image_to_screen(self):
        """Fit the image to screen while maintaining aspect ratio."""
        if not self.original_image:
            return
            
        # Get original dimensions
        img_width, img_height = self.original_image.get_size()
        
        # Calculate aspect ratios
        img_aspect = img_width / img_height
        screen_aspect = self.window_width / self.window_height
        
        # Determine scaling to fill the screen while maintaining aspect ratio
        if img_aspect > screen_aspect:
            # Image is wider - scale based on height
            new_height = self.window_height
            new_width = int(new_height * img_aspect)
        else:
            # Image is taller - scale based on width
            new_width = self.window_width
            new_height = int(new_width / img_aspect)
        
        # Scale the image
        self.base_image = pygame.transform.smoothscale(
            self.original_image, (new_width, new_height)
        )
        
        # Center the image
        self.base_x = (self.window_width - new_width) // 2
        self.base_y = (self.window_height - new_height) // 2
        
    def apply_zoom(self):
        """Apply the Ken Burns zoom effect."""
        if not self.base_image:
            return
            
        # Update zoom factor
        self.zoom_factor += self.zoom_speed * self.zoom_direction
        
        # Reverse direction when limits are reached
        if self.zoom_factor >= self.max_zoom:
            self.zoom_direction = -1
        elif self.zoom_factor <= self.min_zoom:
            self.zoom_direction = 1
        
        # Clamp zoom factor
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))
        
        # Calculate new dimensions
        base_width, base_height = self.base_image.get_size()
        new_width = int(base_width * self.zoom_factor)
        new_height = int(base_height * self.zoom_factor)
        
        # Scale the image
        if new_width > 0 and new_height > 0:
            self.current_image = pygame.transform.smoothscale(
                self.base_image, (new_width, new_height)
            )
            
            # Center the zoomed image
            self.image_x = self.base_x - (new_width - base_width) // 2
            self.image_y = self.base_y - (new_height - base_height) // 2
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Reset zoom
                    self.zoom_factor = 1.0
                    self.zoom_direction = 1
                elif event.key == pygame.K_UP:
                    # Increase zoom speed
                    self.zoom_speed = min(0.01, self.zoom_speed + 0.001)
                elif event.key == pygame.K_DOWN:
                    # Decrease zoom speed
                    self.zoom_speed = max(0.001, self.zoom_speed - 0.001)
    
    def draw(self):
        """Draw the current frame."""
        # Fill background
        self.screen.fill((0, 0, 0))
        
        # Draw the image if available
        if self.current_image:
            self.screen.blit(self.current_image, (self.image_x, self.image_y))
        
        # Draw instructions
        self.draw_instructions()
        
        pygame.display.flip()
    
    def draw_instructions(self):
        """Draw control instructions on screen."""
        font = pygame.font.Font(None, 24)
        instructions = [
            "ESC: Exit",
            "SPACE: Reset zoom",
            "UP/DOWN: Adjust zoom speed",
            f"Zoom: {self.zoom_factor:.2f}x",
            f"Speed: {self.zoom_speed:.3f}"
        ]
        
        y_offset = 10
        for instruction in instructions:
            text = font.render(instruction, True, (255, 255, 255))
            text_rect = text.get_rect()
            text_rect.topleft = (10, y_offset)
            
            # Add background for better readability
            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            
            self.screen.blit(text, text_rect)
            y_offset += 30
    
    def run(self, image_path=None):
        """Main loop for the Ken Burns effect."""
        if image_path and not self.load_image(image_path):
            print("Failed to load image. Please check the path.")
            return
        
        if not image_path:
            # Create a sample gradient image for demonstration
            self.create_sample_image()
        
        while self.running:
            self.handle_events()
            self.apply_zoom()
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
    
    def create_sample_image(self):
        """Create a sample gradient image for demonstration."""
        # Create a sample image with gradient
        sample_width, sample_height = 1920, 1080
        sample_surface = pygame.Surface((sample_width, sample_height))
        
        # Create a colorful gradient
        for y in range(sample_height):
            for x in range(sample_width):
                # Create a radial gradient effect
                center_x, center_y = sample_width // 2, sample_height // 2
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = math.sqrt(center_x**2 + center_y**2)
                
                # Color based on distance and position
                r = int(255 * (1 - distance / max_distance))
                g = int(255 * (x / sample_width))
                b = int(255 * (y / sample_height))
                
                sample_surface.set_at((x, y), (r, g, b))
        
        self.original_image = sample_surface
        self.fit_image_to_screen()


class KenBurnsVideoExporter:
    def __init__(self, output_width=1920, output_height=1080, fps=60, duration_seconds=10):
        """Initialize the Ken Burns video exporter with GPU optimization."""
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.total_frames = fps * duration_seconds
        
        # Animation parameters - slower zoom
        self.max_zoom = 1.2  # Reduced from 1.5 to 1.2 for slower zoom
        self.min_zoom = 1.0
        
        # Always zoom in
        self.zoom_in = True
        
        # Check for GPU support
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_gpu = True
                print("GPU acceleration available and enabled")
            else:
                self.use_gpu = False
                print("GPU not available, using CPU")
        except:
            self.use_gpu = False
            print("GPU support not compiled in OpenCV, using CPU")
        
        print("Effect: ZOOM IN")
        
    def load_and_fit_image(self, image_path):
        """Load and fit image to output dimensions while maintaining aspect ratio."""
        # First try the original image
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"\n🔄 FALLBACK: Original image failed to load: {Path(image_path).name}")
            print(f"   📁 Looking for alternatives in: {Path(image_path).parent}")
            
            # Get all image files in the same directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            alternative_files = []
            
            for ext in image_extensions:
                alternative_files.extend(Path(image_path).parent.glob(f"*{ext}"))
            
            # Filter out the original failed file and system files
            alternative_files = [f for f in alternative_files 
                               if f != Path(image_path) and not f.name.startswith('._')]
            
            if alternative_files:
                print(f"   💡 Found {len(alternative_files)} potential alternatives")
                
                # Try each alternative until one works
                for alt_file in alternative_files:
                    print(f"   🔍 Trying: {alt_file.name}")
                    image = cv2.imread(str(alt_file))
                    if image is not None:
                        print(f"   ✅ Successfully loaded alternative: {alt_file.name}")
                        image_path = alt_file  # Update the path for logging
                        break
                    else:
                        print(f"   ❌ Failed to load: {alt_file.name}")
            
            if image is None:
                print(f"   ❌ No working alternatives found")
                raise ValueError(f"Could not load image or any alternatives in: {Path(image_path).parent}")
        
        # Get original dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate aspect ratios
        img_aspect = img_width / img_height
        output_aspect = self.output_width / self.output_height
        
        # Determine scaling to fill the output while maintaining aspect ratio
        if img_aspect > output_aspect:
            # Image is wider - scale based on height
            new_height = self.output_height
            new_width = int(new_height * img_aspect)
        else:
            # Image is taller - scale based on width
            new_width = self.output_width
            new_height = int(new_width / img_aspect)
        
        # Resize the image using GPU if available
        if self.use_gpu:
            try:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                gpu_resized = cv2.cuda.resize(gpu_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                fitted_image = gpu_resized.download()
            except:
                # Fallback to CPU if GPU fails
                fitted_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            fitted_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Center the image
        base_x = (self.output_width - new_width) // 2
        base_y = (self.output_height - new_height) // 2
        
        return fitted_image, base_x, base_y, new_width, new_height
    
    def generate_ken_burns_frame(self, fitted_image, base_x, base_y, fitted_width, fitted_height, frame_number):
        """Generate a single frame with Ken Burns effect using optimized OpenCV."""
        # Calculate time-based progress (0 to 1) over the full duration
        progress = frame_number / self.total_frames
        
        # Apply zoom in effect (always zoom in now)
        zoom_range = self.max_zoom - self.min_zoom
        zoom_factor = self.min_zoom + (zoom_range * progress)
        
        # Calculate new dimensions
        new_width = int(fitted_width * zoom_factor)
        new_height = int(fitted_height * zoom_factor)
        
        # Resize using GPU if available
        if self.use_gpu:
            try:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(fitted_image)
                gpu_zoomed = cv2.cuda.resize(gpu_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                zoomed_image = gpu_zoomed.download()
            except:
                # Fallback to CPU
                zoomed_image = cv2.resize(fitted_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            zoomed_image = cv2.resize(fitted_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create output frame
        output_frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # Center the zoomed image
        paste_x = base_x - (new_width - fitted_width) // 2
        paste_y = base_y - (new_height - fitted_height) // 2
        
        # Calculate crop and paste coordinates
        src_x_start = max(0, -paste_x)
        src_y_start = max(0, -paste_y)
        src_x_end = min(new_width, src_x_start + self.output_width - max(0, paste_x))
        src_y_end = min(new_height, src_y_start + self.output_height - max(0, paste_y))
        
        dst_x_start = max(0, paste_x)
        dst_y_start = max(0, paste_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Copy the cropped zoomed image to output frame
        if src_x_end > src_x_start and src_y_end > src_y_start:
            output_frame[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                zoomed_image[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return output_frame
    
    def export_video(self, image_path, output_path):
        """Export Ken Burns effect video for a single image."""
        print(f"Processing: {Path(image_path).name}")
        print(f"Output: {output_path}")
        
        # Load and fit the image
        fitted_image, base_x, base_y, fitted_width, fitted_height = self.load_and_fit_image(image_path)
        
        # Use H.264 codec for better compression and speed
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_writer = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            self.fps, 
            (self.output_width, self.output_height)
        )
        
        if not video_writer.isOpened():
            # Fallback to mp4v if H264 fails
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                self.fps, 
                (self.output_width, self.output_height)
            )
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        # Generate and write frames
        progress_step = max(1, self.total_frames // 20)  # Show more frequent progress
        
        for frame_num in range(self.total_frames):
            # Show progress more frequently
            if frame_num % progress_step == 0:
                progress = (frame_num / self.total_frames) * 100
                print(f"  Progress: {progress:.1f}%")
            
            # Generate frame
            frame = self.generate_ken_burns_frame(
                fitted_image, base_x, base_y, fitted_width, fitted_height, frame_num
            )
            
            # Write frame to video
            video_writer.write(frame)
        
        # Release video writer
        video_writer.release()
        
        # Show results with direction-specific info
        direction = "IN" if self.zoom_in else "OUT"
        start_zoom = self.min_zoom if self.zoom_in else self.max_zoom
        end_zoom = self.max_zoom if self.zoom_in else self.min_zoom
        
        print(f"  ✓ Video exported successfully!")
        print(f"  Duration: {self.duration_seconds} seconds")
        print(f"  Resolution: {self.output_width}x{self.output_height}")
        print(f"  FPS: {self.fps}")
        print(f"  Effect: ZOOM {direction} ({start_zoom}x → {end_zoom}x)")


def find_images_in_folder(folder_path):
    """Find all image files in the specified folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return []
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
    
    image_files = []
    try:
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
    except PermissionError:
        print(f"Error: Permission denied accessing '{folder_path}'.")
        return []
    
    return sorted(image_files)


def main():
    """Main function to run the Ken Burns effect."""
    print("Ken Burns Zoom Effect")
    print("====================")
    print("Controls:")
    print("- ESC: Exit")
    print("- SPACE: Reset zoom")
    print("- UP/DOWN: Adjust zoom speed")
    print()
    
    # You can specify an image path here
    image_path = None  # Set to your image path, e.g., "path/to/your/image.jpg"
    
    # Check if an image file exists in the current directory
    common_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    current_dir = Path('.')
    
    for ext in common_extensions:
        image_files = list(current_dir.glob(f'*{ext}'))
        if image_files:
            image_path = str(image_files[0])
            print(f"Found image: {image_path}")
            break
    
    if not image_path:
        print("No image found in current directory. Using sample gradient image.")
    
    # Create and run the effect
    effect = KenBurnsEffect(window_width=1200, window_height=800)
    effect.run(image_path)

if __name__ == "__main__":
    main()

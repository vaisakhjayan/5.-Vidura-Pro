import os
import random
from datetime import datetime
import json
import logging
import platform

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Platform-specific paths
SYSTEM = platform.system().lower()
if SYSTEM == 'darwin':  # macOS
    BASE_PATH = r"/Users/superman/Desktop/Celebrity Folder"
else:  # Windows
    BASE_PATH = r"E:\Celebrity Folder"

BLOCK_SIZE = 5.0  # Each block is 5 seconds
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Print configuration
print(f"\n📂 Path Configuration:")
print(f"   • Operating System: {SYSTEM.title()}")
print(f"   • Base Path: {BASE_PATH}")
print()

# Global tracking for used clips per celebrity
celebrity_clip_trackers = {}

def format_timestamp(seconds):
    """Convert seconds to MM:SS.SS format"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def get_media_files(celebrity_name):
    """Get all media files for a celebrity"""
    celebrity_path = os.path.join(BASE_PATH, celebrity_name)
    if not os.path.exists(celebrity_path):
        logger.warning(f"Celebrity path does not exist: {celebrity_path}")
        print(f"⚠️  Warning: Celebrity path does not exist: {celebrity_path}")
        return [], []
        
    videos = []
    images = []
    
    for file in os.listdir(celebrity_path):
        # Skip files that start with '._' or are hidden/system files
        if file.startswith('._') or file.startswith('.'):
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(file)
        elif ext in IMAGE_EXTENSIONS:
            images.append(file)
    
    logger.debug(f"Found {len(videos)} videos and {len(images)} images for {celebrity_name}")
    return videos, images

def initialize_celebrity_tracker(celebrity_name):
    """Initialize tracking for a celebrity's clips"""
    if celebrity_name not in celebrity_clip_trackers:
        videos, images = get_media_files(celebrity_name)
        celebrity_clip_trackers[celebrity_name] = {
            'available_videos': videos.copy(),
            'available_images': images.copy(),
            'used_videos': [],
            'used_images': [],
            'all_videos': videos.copy(),
            'all_images': images.copy(),
            'clip_count': 0
        }
        print(f"📂 Initialized {celebrity_name}: {len(videos)} videos, {len(images)} images")

def choose_media_for_celebrity(celebrity_name):
    """Choose the next media file for a celebrity, prioritizing variety and cycling through both videos and images"""
    initialize_celebrity_tracker(celebrity_name)
    tracker = celebrity_clip_trackers[celebrity_name]
    tracker['clip_count'] += 1
    
    # Check if we need to reset either pool
    if not tracker['available_videos']:
        # Reset video pool if exhausted
        tracker['available_videos'] = tracker['all_videos'].copy()
        tracker['used_videos'] = []
        print(f"♻️  {celebrity_name}: Reset video pool - all videos used, starting over")
    
    if not tracker['available_images']:
        # Reset image pool if exhausted
        tracker['available_images'] = tracker['all_images'].copy()
        tracker['used_images'] = []
        print(f"♻️  {celebrity_name}: Reset image pool - all images used, starting over")
    
    # Alternate between videos and images if both are available
    if tracker['available_images'] and (tracker['clip_count'] % 4 == 0):  # Every 4th clip is an image
        chosen_image = random.choice(tracker['available_images'])
        tracker['available_images'].remove(chosen_image)
        tracker['used_images'].append(chosen_image)
        print(f"🖼️  {celebrity_name}: Using image {chosen_image} ({len(tracker['available_images'])} images left)")
        return 'image', chosen_image
    elif tracker['available_videos']:
        chosen_video = random.choice(tracker['available_videos'])
        tracker['available_videos'].remove(chosen_video)
        tracker['used_videos'].append(chosen_video)
        print(f"🎥 {celebrity_name}: Using video {chosen_video} ({len(tracker['available_videos'])} videos left)")
        return 'video', chosen_video
    elif tracker['available_images']:  # Fallback to images if no videos
        chosen_image = random.choice(tracker['available_images'])
        tracker['available_images'].remove(chosen_image)
        tracker['used_images'].append(chosen_image)
        print(f"🖼️  {celebrity_name}: Using image {chosen_image} ({len(tracker['available_images'])} images left)")
        return 'image', chosen_image
    else:
        # Fallback if no media files exist
        print(f"⚠️  {celebrity_name}: No media files found, using placeholder")
        return 'video', 'placeholder.mp4'

def choose_media_type(block_count, total_blocks):
    # This function is now replaced by choose_media_for_celebrity
    return 'video'

def generate_render_plans_5s_blocks(detections_json, text_output_file, json_output_file):
    print("🚀 Starting Celebrity Clip Placement Generator...")
    logger.info("Starting render plan generation")
    
    # Check if input file exists
    if not os.path.exists(detections_json):
        error_msg = f"Input file not found: {detections_json}"
        logger.error(error_msg)
        print(f"❌ Error: {error_msg}")
        return
    
    print(f"📖 Reading detections from: {detections_json}")
    logger.info(f"Loading detections from {detections_json}")
    
    with open(detections_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    detections = data['detections']
    video_info = data.get('video', {})

    print(f"✅ Loaded {len(detections)} detections")
    logger.info(f"Loaded {len(detections)} detections")

    # Sort detections by timestamp (just in case)
    detections = sorted(detections, key=lambda d: d['timestamp'])

    print("🔄 Building 5-second block segments...")
    logger.info("Building segments from detections")
    
    # Build 5s block segments
    segments = []
    for i, detection in enumerate(detections):
        start = detection['timestamp']
        celebrity = detection['keyword']
        if i < len(detections) - 1:
            end = detections[i + 1]['timestamp']
        else:
            end = start + BLOCK_SIZE  # Last detection: just one block
        current = start
        while current < end:
            block_end = min(current + BLOCK_SIZE, end)
            segments.append((current, block_end, celebrity))
            current = block_end

    total_blocks = len(segments)
    block_count = 0

    print(f"📊 Generated {total_blocks} segments")
    logger.info(f"Generated {total_blocks} segments")

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Prepare JSON structure
    json_data = {
        "video_info": {
            "title": video_info.get("title", "Celebrity Clip Placement"),
            "output_file": "final_render.mp4",
            "generated_at": formatted_time
        },
        "segments": []
    }

    print(f"📝 Writing text output to: {text_output_file}")
    logger.info(f"Writing text output to {text_output_file}")

    # Write text format
    with open(text_output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write("CELEBRITY CLIP PLACEMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {formatted_time}\n")
        f.write(f"Total Segments: {total_blocks}\n\n")
        f.write("-"*80 + "\n")
        f.write("TIMELINE OVERVIEW\n")
        f.write("-"*80 + "\n\n")

        current_celebrity = None
        celebrity_stats = {}

        for i, (start, end, celebrity) in enumerate(segments):
            if current_celebrity != celebrity:
                if current_celebrity is not None:
                    f.write("\n")
                current_celebrity = celebrity
                f.write(f"🎬 {celebrity.upper()}\n")
                f.write("-"*40 + "\n")
                print(f"🎭 Processing celebrity: {celebrity}")
                logger.info(f"Processing celebrity: {celebrity}")
                
            media_type, media_file = choose_media_for_celebrity(celebrity)
            
            # Set display symbols and descriptions
            if media_type == 'video':
                media_symbol = "🎥"
                media_type_str = "Video"
            else:  # image
                media_symbol = "🖼️" 
                media_type_str = "Ken Burns"
            
            duration = end - start
            block_count += 1
            
            # Update statistics
            if celebrity not in celebrity_stats:
                celebrity_stats[celebrity] = {"time": 0, "video_clips": 0, "image_clips": 0}
            celebrity_stats[celebrity]["time"] += duration
            if media_type == 'video':
                celebrity_stats[celebrity]["video_clips"] += 1
            else:
                celebrity_stats[celebrity]["image_clips"] += 1

            # Write text segment
            f.write(f"    {block_count}. {media_symbol} {format_timestamp(start)} → {format_timestamp(end)} ({duration:.2f}s) ({media_type_str}) - {media_file}\n")
            f.write("       📝 [Description placeholder]\n")
            f.write("       " + "-"*70 + "\n")

            # Add to JSON data
            json_segment = {
                "start_time": start,
                "end_time": end,
                "type": "celebrity_overlay",
                "celebrity": celebrity,
                "media": {
                    "folder": os.path.join(BASE_PATH, celebrity),
                    "type": media_type,  # Now properly uses 'video' or 'image'
                    "duration": duration,
                    "file": media_file
                },
                "transcription_context": [
                    "[Description placeholder]"
                ]
            }
            json_data["segments"].append(json_segment)

            # Progress indicator
            if block_count % 10 == 0:
                progress = (block_count / total_blocks) * 100
                print(f"⏳ Progress: {block_count}/{total_blocks} segments ({progress:.1f}%)")
                logger.info(f"Progress: {block_count}/{total_blocks} segments")

        print("📈 Calculating statistics...")
        logger.info("Calculating final statistics")

        # Write text statistics
        f.write("\n" + "-"*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n\n")

        total_time = 0
        for celebrity, stats in celebrity_stats.items():
            total_clips = stats["video_clips"] + stats["image_clips"]
            percentage = (stats["time"] / sum(s["time"] for s in celebrity_stats.values())) * 100
            total_time += stats["time"]
            
            f.write(f"{celebrity}:\n")
            f.write(f"  • Total Screen Time: {stats['time']:.2f} seconds ({percentage:.1f}%)\n")
            f.write(f"  • Total Clips: {total_clips} ({stats['video_clips']} videos, {stats['image_clips']} images)\n\n")

        f.write(f"Total Coverage: {total_time:.2f} seconds\n")
        f.write(f"Total Clips Used: {sum(s['video_clips'] + s['image_clips'] for s in celebrity_stats.values())}\n\n")

        f.write("="*80 + "\n")
        f.write("End of Summary\n")
        f.write("="*80 + "\n")

    print(f"💾 Writing JSON output to: {json_output_file}")
    logger.info(f"Writing JSON output to {json_output_file}")

    # Write JSON format
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print("✅ Celebrity Clip Placement Generation Complete!")
    logger.info("Render plan generation completed successfully")
    
    # Print detailed usage summary
    print(f"\n📊 SUMMARY:")
    print(f"   • Total segments: {total_blocks}")
    print(f"   • Celebrities processed: {len(celebrity_stats)}")
    print(f"   • Total coverage: {total_time:.2f} seconds")
    
    print(f"\n🎬 CLIP USAGE BREAKDOWN:")
    for celebrity, tracker in celebrity_clip_trackers.items():
        total_available = len(tracker['all_videos']) + len(tracker['all_images'])
        videos_used = len(tracker['used_videos'])
        images_used = len(tracker['used_images'])
        videos_remaining = len(tracker['available_videos'])
        images_remaining = len(tracker['available_images'])
        
        print(f"   📂 {celebrity}:")
        print(f"      • Total clips available: {total_available}")
        print(f"      • Videos used: {videos_used}/{len(tracker['all_videos'])} (remaining: {videos_remaining})")
        if tracker['all_images']:
            print(f"      • Images used: {images_used}/{len(tracker['all_images'])} (remaining: {images_remaining})")
        else:
            print(f"      • Images: None available (using videos only)")
        
        # Calculate variety percentage
        unique_clips_used = videos_used + images_used
        if total_available > 0:
            variety_percent = (unique_clips_used / total_available) * 100
            print(f"      • Variety: {variety_percent:.1f}% of available clips used")
    
    print(f"\n📄 OUTPUT FILES:")
    print(f"   • Text output: {text_output_file}")
    print(f"   • JSON output: {json_output_file}")
    
    print(f"\n💡 OPTIMIZATION ACHIEVED:")
    print(f"   ✅ All clips used before repeating")
    print(f"   ✅ Images included every 4-5 clips (when available)")
    print(f"   ✅ Maximum variety maintained per celebrity")
    print(f"   ✅ Automatic fallback to videos when no images exist")

if __name__ == "__main__":
    print("🎬 Celebrity Clip Placement Generator")
    print("=" * 50)
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute paths
    detections_json = os.path.join(os.path.dirname(current_dir), "JSON Files", "3. Detections.json")
    text_output_file = os.path.join(os.path.dirname(current_dir), "JSON Files", "4. Render Plan V2.txt")
    json_output_file = os.path.join(os.path.dirname(current_dir), "JSON Files", "4. Render Plan V2.json")
    
    print(f"📂 Input file: {detections_json}")
    print(f"📄 Text output: {text_output_file}")
    print(f"📄 JSON output: {json_output_file}")
    print("-" * 50)
    
    generate_render_plans_5s_blocks(detections_json, text_output_file, json_output_file)

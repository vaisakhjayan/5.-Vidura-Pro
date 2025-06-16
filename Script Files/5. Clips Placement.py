import os
import random
from datetime import datetime
import json
import logging
import platform
from collections import Counter
import re

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
    INTELLIGENT_CLIPS_PATH = r"/Users/superman/Desktop/Intelligent Clip"
else:  # Windows
    BASE_PATH = r"E:\Celebrity Folder"
    INTELLIGENT_CLIPS_PATH = r"E:\Intelligent Clip"

BLOCK_SIZE = 5.0  # Each block is 5 seconds
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

class IntelligentClipSuggestion:
    def __init__(self, start_time, end_time, folder, description, similarity):
        self.start_time = start_time
        self.end_time = end_time
        self.folder = folder
        self.description = description
        self.similarity = similarity

def read_intelligent_suggestions(file_path="3 Large Output.txt"):
    """Read and parse intelligent clip suggestions from the output file."""
    suggestions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract matches using regex
        matches = re.finditer(
            r"Timestamp: (\d+\.\d+)s - (\d+\.\d+)s.*?"
            r"Folder: ([^\n]+).*?"
            r"Description: ([^\n]+).*?"
            r"Similarity Score: (\d+\.\d+)",
            content, re.DOTALL
        )
        
        for match in matches:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            folder = match.group(3).strip()
            description = match.group(4).strip()
            similarity = float(match.group(5))
            
            suggestions.append(IntelligentClipSuggestion(
                start_time=start_time,
                end_time=end_time,
                folder=folder,
                description=description,
                similarity=similarity
            ))
            
        print(f"📖 Loaded {len(suggestions)} intelligent clip suggestions")
        return suggestions
    except Exception as e:
        print(f"⚠️ Error reading intelligent suggestions: {e}")
        return []

def get_intelligent_clip_media(folder):
    """Get media files from the intelligent clips folder."""
    folder_path = os.path.join(INTELLIGENT_CLIPS_PATH, folder)
    if not os.path.exists(folder_path):
        logger.warning(f"Intelligent clip folder does not exist: {folder_path}")
        print(f"⚠️  Intelligent clip folder not found: {folder_path}")
        return None, None, None
        
    media_files = []
    for file in os.listdir(folder_path):
        if file.startswith('._') or file.startswith('.'):
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext in VIDEO_EXTENSIONS or ext in IMAGE_EXTENSIONS:
            media_files.append((file, ext))
    
    if not media_files:
        print(f"⚠️  No media files found in: {folder_path}")
        return None, None, None
        
    # Sort files to ensure consistent selection
    media_files.sort()
    
    videos = [(f, ext) for f, ext in media_files if ext in VIDEO_EXTENSIONS]
    images = [(f, ext) for f, ext in media_files if ext in IMAGE_EXTENSIONS]
    
    # Try to get two different clips
    clips = []
    
    # First try to get two videos
    if len(videos) >= 2:
        print(f"✨ Found two intelligent video clips")
        clips = [('video', os.path.join(folder_path, videos[0][0])), 
                ('video', os.path.join(folder_path, videos[1][0]))]
    # Then try one video + one image
    elif len(videos) == 1 and images:
        print(f"✨ Found one intelligent video clip and one image")
        clips = [('video', os.path.join(folder_path, videos[0][0])), 
                ('image', os.path.join(folder_path, images[0][0]))]
    # Then try two images
    elif len(images) >= 2:
        print(f"✨ Found two intelligent images")
        clips = [('image', os.path.join(folder_path, images[0][0])), 
                ('image', os.path.join(folder_path, images[1][0]))]
    # Finally try single video or image
    elif videos:
        print(f"✨ Found single intelligent video clip (will be used twice)")
        clips = [('video', os.path.join(folder_path, videos[0][0]))] * 2
    elif images:
        print(f"✨ Found single intelligent image (will be used twice)")
        clips = [('image', os.path.join(folder_path, images[0][0]))] * 2
    else:
        return None, None, None
        
    return clips[0][0], clips[0][1], clips[1][1]  # Returns: type, first_file, second_file

# Print configuration
print(f"\n📂 Path Configuration:")
print(f"   • Operating System: {SYSTEM.title()}")
print(f"   • Base Path: {BASE_PATH}")
print(f"   • Intelligent Clips Path: {INTELLIGENT_CLIPS_PATH}")
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
            videos.append(os.path.join(celebrity_path, file))  # Store full path
        elif ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(celebrity_path, file))  # Store full path
    
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
        print(f"🖼️  {celebrity_name}: Using image {os.path.basename(chosen_image)} ({len(tracker['available_images'])} images left)")
        return 'image', chosen_image
    elif tracker['available_videos']:
        chosen_video = random.choice(tracker['available_videos'])
        tracker['available_videos'].remove(chosen_video)
        tracker['used_videos'].append(chosen_video)
        print(f"🎥 {celebrity_name}: Using video {os.path.basename(chosen_video)} ({len(tracker['available_videos'])} videos left)")
        return 'video', chosen_video
    elif tracker['available_images']:  # Fallback to images if no videos
        chosen_image = random.choice(tracker['available_images'])
        tracker['available_images'].remove(chosen_image)
        tracker['used_images'].append(chosen_image)
        print(f"🖼️  {celebrity_name}: Using image {os.path.basename(chosen_image)} ({len(tracker['available_images'])} images left)")
        return 'image', chosen_image
    else:
        # Fallback if no media files exist
        print(f"⚠️  {celebrity_name}: No media files found, using placeholder")
        return 'video', 'placeholder.mp4'

def choose_media_type(block_count, total_blocks):
    # This function is now replaced by choose_media_for_celebrity
    return 'video'

def get_top_celebrities(detections, top_n=2):
    """Get the top N most frequently mentioned celebrities from detections"""
    celebrity_counts = Counter(d['keyword'] for d in detections)
    return [celeb for celeb, _ in celebrity_counts.most_common(top_n)]

def generate_initial_segments(first_detection_time, top_celebrities):
    """Generate segments to fill the gap before first detection"""
    if first_detection_time <= 0:
        return []
        
    segments = []
    current_time = 0
    
    while current_time < first_detection_time:
        block_end = min(current_time + BLOCK_SIZE, first_detection_time)
        duration = block_end - current_time
        if duration < 0.5:  # Skip if remaining duration is too short
            break
            
        # Randomly choose from top celebrities
        celebrity = random.choice(top_celebrities)
        segments.append((current_time, block_end, celebrity))
        current_time = block_end
        
    return segments

def generate_render_plans_5s_blocks(detections_json, text_output_file, json_output_file):
    print("🚀 Starting Celebrity Clip Placement Generator...")
    logger.info("Starting render plan generation")
    
    # Load intelligent suggestions first
    intelligent_suggestions = read_intelligent_suggestions()
    print(f"✨ Loaded {len(intelligent_suggestions)} intelligent clip suggestions")
    
    # Create a set of ranges that are reserved for intelligent clips
    reserved_ranges = []
    for suggestion in intelligent_suggestions:
        reserved_ranges.append((suggestion.start_time, suggestion.end_time))
    reserved_ranges.sort(key=lambda x: x[0])
    
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

    # Sort detections by timestamp
    detections = sorted(detections, key=lambda d: d['timestamp'])
    
    # Get top celebrities and generate initial segments
    top_celebrities = get_top_celebrities(detections)
    first_detection_time = detections[0]['timestamp'] if detections else 0
    initial_segments = generate_initial_segments(first_detection_time, top_celebrities)
    
    # Build segments, incorporating intelligent suggestions
    segments = initial_segments.copy()
    current_suggestion_idx = 0
    
    def is_time_in_reserved_range(time_point):
        """Check if a time point falls within any reserved range"""
        for start, end in reserved_ranges:
            if start <= time_point < end:
                return True
        return False
    
    def get_next_available_time(current_time):
        """Get the next available time that's not in a reserved range"""
        while any(start <= current_time < end for start, end in reserved_ranges):
            # Find the range we're in and skip to its end
            for start, end in reserved_ranges:
                if start <= current_time < end:
                    current_time = end
                    break
        return current_time
    
    # First, add all intelligent suggestions as segments
    for suggestion in intelligent_suggestions:
        segments.append((
            suggestion.start_time,
            suggestion.end_time,  # This will be a 10-second duration
            detections[0]['keyword'],  # Use the first detected celebrity as default
            suggestion
        ))
    
    # Then add regular segments, skipping over reserved ranges
    for i, detection in enumerate(detections):
        start = detection['timestamp']
        celebrity = detection['keyword']
        
        # Skip if we're in a reserved range
        start = get_next_available_time(start)
        
        if i < len(detections) - 1:
            end = detections[i + 1]['timestamp']
        else:
            end = start + BLOCK_SIZE
        
        current = start
        while current < end:
            # Skip if we're in a reserved range
            current = get_next_available_time(current)
            if current >= end:
                break
                
            block_end = min(current + BLOCK_SIZE, end)
            # Check if this block would overlap with a reserved range
            for r_start, r_end in reserved_ranges:
                if current < r_start < block_end:
                    block_end = r_start
                    break
            
            if block_end > current:  # Only add if we have a valid duration
                segments.append((current, block_end, celebrity))
            current = block_end

    # Sort segments by start time and fix any overlaps
    segments.sort(key=lambda x: x[0])
    for i in range(len(segments)-1):
        if segments[i][1] > segments[i+1][0]:
            print(f"⚠️ Fixing overlap between segments at {format_timestamp(segments[i][0])} and {format_timestamp(segments[i+1][0])}")
            # Adjust the end time of the current segment to match the start of the next
            segments[i] = (segments[i][0], segments[i+1][0], segments[i][2], segments[i][3] if len(segments[i]) > 3 else None)

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

        for i, segment in enumerate(segments):
            start, end = segment[0], segment[1]
            celebrity = segment[2]
            is_intelligent_suggestion = len(segment) > 3
            
            if current_celebrity != celebrity:
                if current_celebrity is not None:
                    f.write("\n")
                current_celebrity = celebrity
                f.write(f"🎬 {celebrity.upper()}\n")
                f.write("-"*40 + "\n")
                
            if is_intelligent_suggestion:
                suggestion = segment[3]
                media_type, first_file, second_file = get_intelligent_clip_media(suggestion.folder)
                if first_file and second_file:
                    media_symbol = "✨" if media_type == 'video' else "💫"
                    media_type_str = "Intelligent Video" if media_type == 'video' else "Intelligent Image"
                    
                    # Split the 10-second segment into two 5-second parts
                    mid_time = start + (end - start) / 2
                    block_count += 1
                    
                    # Calculate actual durations
                    first_duration = mid_time - start
                    second_duration = end - mid_time
                    
                    # First part
                    json_segment = {
                        "start_time": start,
                        "end_time": mid_time,
                        "type": "intelligent_clip",
                        "celebrity": celebrity,
                        "media": {
                            "folder": os.path.dirname(first_file),
                            "type": media_type,
                            "duration": first_duration,
                            "file": os.path.basename(first_file)
                        },
                        "transcription_context": [suggestion.description],
                        "similarity_score": suggestion.similarity,
                        "intelligent_folder": suggestion.folder
                    }
                    json_data["segments"].append(json_segment)
                    
                    # Second part
                    json_segment = {
                        "start_time": mid_time,
                        "end_time": end,
                        "type": "intelligent_clip",
                        "celebrity": celebrity,
                        "media": {
                            "folder": os.path.dirname(second_file),
                            "type": media_type,
                            "duration": second_duration,
                            "file": os.path.basename(second_file)
                        },
                        "transcription_context": [suggestion.description],
                        "similarity_score": suggestion.similarity,
                        "intelligent_folder": suggestion.folder
                    }
                    json_data["segments"].append(json_segment)
                    
                    # Write text segments with actual durations
                    f.write(f"    {block_count}a. {media_symbol} {format_timestamp(start)} → {format_timestamp(mid_time)} ({first_duration:.2f}s) ({media_type_str}) - {os.path.basename(first_file)}\n")
                    f.write(f"       📝 {suggestion.description} (Part 1)\n")
                    f.write("       " + "-"*70 + "\n")
                    f.write(f"    {block_count}b. {media_symbol} {format_timestamp(mid_time)} → {format_timestamp(end)} ({second_duration:.2f}s) ({media_type_str}) - {os.path.basename(second_file)}\n")
                    f.write(f"       📝 {suggestion.description} (Part 2)\n")
                    f.write("       " + "-"*70 + "\n")
                else:
                    # Fallback to regular clip selection if intelligent clips not found
                    media_type, media_file = choose_media_for_celebrity(celebrity)
                    media_symbol = "🎥" if media_type == 'video' else "🖼️"
                    media_type_str = "Video" if media_type == 'video' else "Ken Burns"
                    block_count += 1
                    
                    json_segment = {
                        "start_time": start,
                        "end_time": end,
                        "type": "celebrity_overlay",
                        "celebrity": celebrity,
                        "media": {
                            "folder": os.path.dirname(media_file),
                            "type": media_type,
                            "duration": end - start,
                            "file": os.path.basename(media_file)
                        },
                        "transcription_context": ["[Description placeholder]"]
                    }
                    json_data["segments"].append(json_segment)
                    
                    f.write(f"    {block_count}. {media_symbol} {format_timestamp(start)} → {format_timestamp(end)} ({end - start:.2f}s) ({media_type_str}) - {os.path.basename(media_file)}\n")
                    f.write(f"       📝 [Description placeholder]\n")
                    f.write("       " + "-"*70 + "\n")
            else:
                # Regular clip handling
                media_type, media_file = choose_media_for_celebrity(celebrity)
                media_symbol = "🎥" if media_type == 'video' else "🖼️"
                media_type_str = "Video" if media_type == 'video' else "Ken Burns"
                block_count += 1
                
                json_segment = {
                    "start_time": start,
                    "end_time": end,
                    "type": "celebrity_overlay",
                    "celebrity": celebrity,
                    "media": {
                        "folder": os.path.dirname(media_file),
                        "type": media_type,
                        "duration": end - start,
                        "file": os.path.basename(media_file)
                    },
                    "transcription_context": ["[Description placeholder]"]
                }
                json_data["segments"].append(json_segment)
                
                f.write(f"    {block_count}. {media_symbol} {format_timestamp(start)} → {format_timestamp(end)} ({end - start:.2f}s) ({media_type_str}) - {os.path.basename(media_file)}\n")
                f.write(f"       📝 [Description placeholder]\n")
                f.write("       " + "-"*70 + "\n")

            # Update statistics
            if celebrity not in celebrity_stats:
                celebrity_stats[celebrity] = {"time": 0, "video_clips": 0, "image_clips": 0, "intelligent_clips": 0}
            celebrity_stats[celebrity]["time"] += end - start
            if is_intelligent_suggestion:
                celebrity_stats[celebrity]["intelligent_clips"] += 1
            elif media_type == 'video':
                celebrity_stats[celebrity]["video_clips"] += 1
            else:
                celebrity_stats[celebrity]["image_clips"] += 1

            # Progress indicator
            if block_count % 10 == 0:
                progress = (block_count / total_blocks) * 100
                print(f"⏳ Progress: {block_count}/{total_blocks} segments ({progress:.1f}%)")

        # Write statistics
        f.write("\n" + "-"*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n\n")

        total_time = 0
        for celebrity, stats in celebrity_stats.items():
            total_clips = stats["video_clips"] + stats["image_clips"] + stats["intelligent_clips"]
            percentage = (stats["time"] / sum(s["time"] for s in celebrity_stats.values())) * 100
            total_time += stats["time"]
            
            f.write(f"{celebrity}:\n")
            f.write(f"  • Total Screen Time: {stats['time']:.2f} seconds ({percentage:.1f}%)\n")
            f.write(f"  • Total Clips: {total_clips}\n")
            f.write(f"    - Intelligent Clips: {stats['intelligent_clips']}\n")
            f.write(f"    - Regular Videos: {stats['video_clips']}\n")
            f.write(f"    - Ken Burns Images: {stats['image_clips']}\n\n")

        f.write(f"Total Coverage: {total_time:.2f} seconds\n")
        f.write(f"Total Clips Used: {sum(s['video_clips'] + s['image_clips'] + s['intelligent_clips'] for s in celebrity_stats.values())}\n\n")

        f.write("="*80 + "\n")
        f.write("End of Summary\n")
        f.write("="*80 + "\n")

    # Write JSON output
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)

    print("✅ Celebrity Clip Placement Generation Complete!")
    logger.info("Render plan generation completed successfully")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"   • Total segments: {total_blocks}")
    print(f"   • Intelligent clips used: {sum(s['intelligent_clips'] for s in celebrity_stats.values())}")
    print(f"   • Regular clips used: {sum(s['video_clips'] + s['image_clips'] for s in celebrity_stats.values())}")
    print(f"   • Total coverage: {total_time:.2f} seconds")
    
    print(f"\n📄 OUTPUT FILES:")
    print(f"   • Text output: {text_output_file}")
    print(f"   • JSON output: {json_output_file}")

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

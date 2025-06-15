import json
import sys
import os
from pathlib import Path
import importlib.util
import shutil
import re
import requests
import platform
import subprocess

# Platform-specific configuration
SYSTEM = platform.system().lower()
USE_CUDA = SYSTEM == 'windows'  # Only use CUDA on Windows

# Add the Other Files directory to the Python path so we can import from it
current_dir = Path(__file__).parent
other_files_dir = current_dir.parent / "Other Files"
json_files_dir = current_dir.parent / "JSON Files"

# Configure FFmpeg settings based on platform
def get_ffmpeg_config():
    """Get platform-specific FFmpeg configuration."""
    if SYSTEM == 'windows':
        return {
            'hwaccel': 'cuda',
            'video_codec': 'h264_nvenc',
            'preset': 'p4',
            'pixel_format': 'yuv420p'
        }
    elif SYSTEM == 'darwin':  # macOS
        return {
            'hwaccel': 'videotoolbox',  # macOS hardware acceleration
            'video_codec': 'h264_videotoolbox',  # macOS hardware encoder
            'preset': 'medium',
            'pixel_format': 'yuv420p'
        }
    else:  # Linux or other
        return {
            'hwaccel': None,
            'video_codec': 'libx264',
            'preset': 'medium',
            'pixel_format': 'yuv420p'
        }

FFMPEG_CONFIG = get_ffmpeg_config()

# Print platform info
print(f"\n🖥️  Platform Configuration:")
print(f"   • Operating System: {SYSTEM.title()}")
print(f"   • CUDA Enabled: {'Yes' if USE_CUDA else 'No'}")
print(f"   • FFmpeg Hardware Acceleration: {FFMPEG_CONFIG['hwaccel'] or 'None'}")
print(f"   • Video Codec: {FFMPEG_CONFIG['video_codec']}")
print()

# Add the Script Files directory to Python path for imports
script_files_dir = Path(__file__).parent
if str(script_files_dir) not in sys.path:
    sys.path.append(str(script_files_dir))

# =============================================================================
# NOTION INTEGRATION FUNCTIONS
# =============================================================================

def update_notion_checkbox(video_title=None):
    """Update the 'Video Edited' checkbox in Notion database."""
    
    # Notion configuration
    NOTION_TOKEN = "ntn_cC7520095381SElmcgTOADYsGnrABFn2ph1PrcaGSst2dv"
    DATABASE_ID = "1a402cd2c14280909384df6c898ddcb3"
    CHECKBOX_PROPERTY = "Video Edited"
    
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    try:
        print(f"\n🔄 Updating Notion database...")
        
        # Query the database to find the page
        query_url = f"https://api.notion.com/v1/databases/{DATABASE_ID}/query"
        
        # First, get all pages without filtering to find the right one
        query_data = {}
        
        print(f"   🔍 Querying database for all pages...")
        response = requests.post(query_url, headers=headers, json=query_data)
        
        if response.status_code != 200:
            print(f"   ❌ Failed to query database: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        results = response.json().get("results", [])
        
        if not results:
            print(f"   ❌ No pages found in database")
            return False
        
        print(f"   📄 Found {len(results)} pages in database")
        
        # Find the best matching page
        target_page = None
        target_page_title = ""
        
        # If we have a video title, try to find a matching page
        if video_title:
            print(f"   🔍 Searching for video: {video_title}")
            video_title_lower = video_title.lower()
            
            for page in results:
                if "properties" in page and "New Title" in page["properties"]:
                    # Look specifically for the "New Title" property
                    title_prop = page["properties"]["New Title"]
                    if title_prop.get("type") == "title" and title_prop.get("title"):
                        page_title = "".join([text.get("plain_text", "") for text in title_prop["title"]])
                        if page_title and video_title_lower in page_title.lower():
                            target_page = page
                            target_page_title = page_title
                            print(f"   ✅ Found matching page: {page_title}")
                            break
        
        # If no specific match found, use the first page (or most recent)
        if not target_page:
            target_page = results[0]  # Use first page as fallback
            
            # Get the title from the "New Title" property
            if "properties" in target_page and "New Title" in target_page["properties"]:
                title_prop = target_page["properties"]["New Title"]
                if title_prop.get("type") == "title" and title_prop.get("title"):
                    target_page_title = "".join([text.get("plain_text", "") for text in title_prop["title"]])
            
            if video_title:
                print(f"   ⚠️  No exact match found, using: {target_page_title or 'first page'}")
            else:
                print(f"   ✅ Using page: {target_page_title or 'first page'}")
        
        page_id = target_page["id"]
        
        # Update the checkbox
        update_url = f"https://api.notion.com/v1/pages/{page_id}"
        update_data = {
            "properties": {
                CHECKBOX_PROPERTY: {
                    "checkbox": True
                }
            }
        }
        
        response = requests.patch(update_url, headers=headers, json=update_data)
        
        if response.status_code == 200:
            print(f"   ✅ Successfully ticked '{CHECKBOX_PROPERTY}' checkbox!")
            print(f"   📋 Page: {target_page_title or page_id}")
            return True
        else:
            print(f"   ❌ Failed to update checkbox: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error updating Notion: {e}")
        return False

# =============================================================================
# HELPER FUNCTIONS FOR VIDEO TITLE AND FOLDER MANAGEMENT
# =============================================================================

def load_selected_video_info():
    """Load the selected video information from JSON file."""
    selected_video_path = json_files_dir / "1. Selected Video.json"
    
    try:
        with open(selected_video_path, 'r', encoding='utf-8') as f:
            video_info = json.load(f)
        print(f"✓ Loaded selected video info: {selected_video_path}")
        return video_info
    except FileNotFoundError:
        print(f"Warning: Selected video info not found at {selected_video_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in selected video info: {e}")
        return None

def sanitize_folder_name(title):
    """Sanitize video title to create a valid, clean folder name."""
    if not title:
        return "Unknown Video"
    
    # Remove or replace invalid characters for Windows folder names
    # Invalid characters: < > : " | ? * \ /
    sanitized = re.sub(r'[<>:"|?*\\/]', '', title)
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Clean up common issues
    # Remove excessive capitalization and make title case
    words = sanitized.split()
    cleaned_words = []
    
    for word in words:
        # Skip empty words
        if not word:
            continue
            
        # Handle all-caps words by making them title case, except for common acronyms
        if word.isupper() and len(word) > 1:
            # Keep common acronyms uppercase
            if word in ['UK', 'US', 'USA', 'TV', 'BBC', 'CNN', 'FBI', 'CIA', 'NASA', 'CEO', 'AI', 'IT']:
                cleaned_words.append(word)
            else:
                cleaned_words.append(word.title())
        else:
            cleaned_words.append(word)
    
    sanitized = ' '.join(cleaned_words)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length to avoid path issues (Windows has 260 char limit)
    # Keep it reasonable but allow for longer, more descriptive names
    if len(sanitized) > 80:
        # Try to cut at a word boundary
        words = sanitized[:80].split()
        if len(words) > 1:
            words.pop()  # Remove the last potentially cut word
            sanitized = ' '.join(words)
        else:
            sanitized = sanitized[:80]
    
    # If empty after sanitization, use default
    if not sanitized:
        sanitized = "Unknown Video"
    
    return sanitized

def create_project_output_folder():
    """Create and return the project-specific output folder based on video title."""
    # Platform-specific base folder
    if SYSTEM == 'darwin':  # macOS
        base_folder = Path("/Users/superman/Desktop/Clips Assembled")
    else:  # Windows
        base_folder = Path(r"E:\Temp")
    
    # Load video info
    video_info = load_selected_video_info()
    if video_info and 'title' in video_info:
        title = video_info['title']
        sanitized_title = sanitize_folder_name(title)
        project_folder = base_folder / sanitized_title
        
        print(f"📁 Creating project folder based on video title:")
        print(f"   Original title: {title}")
        print(f"   Sanitized name: {sanitized_title}")
        print(f"   Full path: {project_folder}")
    else:
        # Fallback if no video info available
        project_folder = base_folder / "Unknown_Video_Project"
        print(f"⚠️  No video title found, using fallback folder: {project_folder}")
    
    # Create the folder
    try:
        project_folder.mkdir(parents=True, exist_ok=True)
        print(f"✓ Project folder created/verified: {project_folder}")
        return str(project_folder)
    except Exception as e:
        print(f"✗ Failed to create project folder: {e}")
        print(f"   Falling back to base temp folder: {base_folder}")
        base_folder.mkdir(exist_ok=True)
        return str(base_folder)

# =============================================================================
# EFFECTS CONFIGURATION
# =============================================================================
class EffectsConfig:
    """Configuration settings for effects processing."""
    
    # Split Screen Effect (Conflict Resolution)
    SPLIT_SCREEN_ENABLED = True            # Enable/disable split screen conflict resolution
    SPLIT_SCREEN_TIME_THRESHOLD = 3.0  # Seconds - celebrities within this time get split screen
    SPLIT_SCREEN_DURATION = 5.0        # Duration for split screen segments
    
    # Ken Burns Effect (Images)
    KENBURNS_ALWAYS_APPLY = True  # Apply to all images
    
    # Composite Effect (Videos)
    COMPOSITE_APPLY_INTERVAL = 3  # Default - will be overridden by Notion value below
    COMPOSITE_START_INDEX = 0     # Start applying from this clip index (0-based)
    
    # Output Settings - will be dynamically set based on video title
    OUTPUT_FOLDER = create_project_output_folder()  # Dynamic folder creation
    OUTPUT_WIDTH = 1920
    OUTPUT_HEIGHT = 1080
    OUTPUT_FPS = 24
    
    # Composite Effect Settings
    COMPOSITE_SCALE_FACTOR = 0.71  # 71% of original size

# Try to import the composite configuration from Notion
try:
    # Import and initialize Notion configuration
    transcription_path = Path(__file__).parent / "3. Transcription.py"
    if transcription_path.exists():
        spec = importlib.util.spec_from_file_location("transcription", transcription_path)
        transcription_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transcription_module)
        
        # Initialize configuration from Notion
        dry_run_mode, COMPOSITE_APPLY_INTERVAL_FROM_NOTION, SPLIT_SCREEN_ENABLED_FROM_NOTION = transcription_module.initialize_notion_config()
        print(f"📡 Loaded configuration from Notion:")
        print(f"   🧪 Dry Run Mode: {dry_run_mode}")
        print(f"   🎬 Composite Apply Interval: {COMPOSITE_APPLY_INTERVAL_FROM_NOTION}")
        print(f"   📱 Split Screen Effect: {'Enabled' if SPLIT_SCREEN_ENABLED_FROM_NOTION else 'Disabled'}")
        
        # Also fetch background configuration if Clip Composite module is available
        try:
            clip_composite_path = Path(__file__).parent.parent / "Other Files" / "Clip Composite.py"
            if clip_composite_path.exists():
                spec = importlib.util.spec_from_file_location("clip_composite", clip_composite_path)
                clip_composite_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(clip_composite_module)
                
                background_name = clip_composite_module.get_background_for_composite()
                print(f"   🎨 Background For Composite: {background_name}")
        except Exception as bg_error:
            print(f"   ⚠️  Could not fetch background configuration: {bg_error}")
            
    else:
        COMPOSITE_APPLY_INTERVAL_FROM_NOTION = 3
        SPLIT_SCREEN_ENABLED_FROM_NOTION = True
        print("⚠️  Transcription module not found, using defaults: Composite=3, Split Screen=Enabled")
        
except Exception as e:
    COMPOSITE_APPLY_INTERVAL_FROM_NOTION = 3
    SPLIT_SCREEN_ENABLED_FROM_NOTION = True
    print(f"⚠️  Error loading Notion configuration: {e}, using defaults: Composite=3, Split Screen=Enabled")

config = EffectsConfig()

# Override composite interval with Notion value if available
if 'COMPOSITE_APPLY_INTERVAL_FROM_NOTION' in locals():
    config.COMPOSITE_APPLY_INTERVAL = COMPOSITE_APPLY_INTERVAL_FROM_NOTION
    print(f"✓ Updated config.COMPOSITE_APPLY_INTERVAL to {COMPOSITE_APPLY_INTERVAL_FROM_NOTION} from Notion")

# Override split screen setting with Notion value if available
if 'SPLIT_SCREEN_ENABLED_FROM_NOTION' in locals():
    config.SPLIT_SCREEN_ENABLED = SPLIT_SCREEN_ENABLED_FROM_NOTION
    print(f"✓ Updated config.SPLIT_SCREEN_ENABLED to {SPLIT_SCREEN_ENABLED_FROM_NOTION} from Notion")

def import_module_from_path(module_name, file_path):
    """Import a module from a specific file path."""
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find '{file_path.name}' at {file_path}")
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
        
    except Exception as e:
        print(f"Error importing {module_name} from {file_path}: {e}")
        return None

def load_render_plan():
    """Load the render plan JSON file."""
    render_plan_path = json_files_dir / "4. Render Plan V2.json"
    
    try:
        with open(render_plan_path, 'r', encoding='utf-8') as f:
            render_plan = json.load(f)
        print(f"✓ Loaded render plan: {render_plan_path}")
        return render_plan
    except FileNotFoundError:
        print(f"Error: Render plan not found at {render_plan_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in render plan: {e}")
        return None

def detect_overlapping_segments(segments):
    """Detect segments that are within the time threshold and group them for split screen processing."""
    print(f"\n🔍 Detecting segments within {config.SPLIT_SCREEN_TIME_THRESHOLD} seconds of each other...")
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x['start_time'])
    
    print(f"📋 Analyzing {len(sorted_segments)} segments:")
    for i, seg in enumerate(sorted_segments):
        print(f"   {i+1}. {seg['celebrity']}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
    
    overlapping_groups = []
    processed_indices = set()
    
    for i, segment1 in enumerate(sorted_segments):
        if i in processed_indices:
            continue
        
        print(f"\n🔍 Checking segment {i+1}: {segment1['celebrity']} at {segment1['start_time']:.2f}s")
        
        # Start with the current segment
        current_group = [segment1]
        current_indices = {i}
        
        # Look for segments within the time threshold
        for j, segment2 in enumerate(sorted_segments):
            if j <= i or j in processed_indices:
                continue
                
            # Calculate the time gap between segments
            time_gap = abs(segment1['start_time'] - segment2['start_time'])
            
            print(f"   🔢 Gap to {segment2['celebrity']} ({segment2['start_time']:.2f}s): {time_gap:.2f}s")
            
            # Only group if segments are within the configured threshold
            if time_gap <= config.SPLIT_SCREEN_TIME_THRESHOLD:
                current_group.append(segment2)
                current_indices.add(j)
                print(f"   ✅ GROUPING: {segment1['celebrity']} with {segment2['celebrity']} - Gap: {time_gap:.2f}s ≤ {config.SPLIT_SCREEN_TIME_THRESHOLD}s")
            else:
                print(f"   ❌ SKIPPING: {segment1['celebrity']} and {segment2['celebrity']} - Gap: {time_gap:.2f}s > {config.SPLIT_SCREEN_TIME_THRESHOLD}s")
        
        # Only create a group if we have multiple segments AND they're truly within threshold
        if len(current_group) > 1:
            # Double-check: verify all segments in group are within threshold of each other
            valid_group = True
            for group_seg1 in current_group:
                for group_seg2 in current_group:
                    if group_seg1 != group_seg2:
                        gap = abs(group_seg1['start_time'] - group_seg2['start_time'])
                        if gap > config.SPLIT_SCREEN_TIME_THRESHOLD:
                            print(f"   ⚠️  INVALID GROUP: {group_seg1['celebrity']} and {group_seg2['celebrity']} gap {gap:.2f}s > {config.SPLIT_SCREEN_TIME_THRESHOLD}s")
                            valid_group = False
                            break
                if not valid_group:
                    break
            
            if valid_group:
                group_start = min(seg['start_time'] for seg in current_group)
                group_end = max(seg['end_time'] for seg in current_group)
                
                overlapping_groups.append({
                    'segments': current_group,
                    'indices': current_indices,
                    'start_time': group_start,
                    'end_time': group_end
                })
                processed_indices.update(current_indices)
                
                print(f"   ✅ VALID GROUP CREATED with {len(current_group)} segments:")
                for seg in current_group:
                    print(f"      • {seg['celebrity']}: {seg['start_time']:.2f}-{seg['end_time']:.2f}s")
            else:
                print(f"   ❌ GROUP REJECTED: Not all segments within {config.SPLIT_SCREEN_TIME_THRESHOLD}s of each other")
        else:
            print(f"   ❌ NO GROUP: {segment1['celebrity']} has no other segments within {config.SPLIT_SCREEN_TIME_THRESHOLD}s")
    
    print(f"\n📊 FINAL RESULT: {len(overlapping_groups)} valid groups for split screen processing")
    if len(overlapping_groups) == 0:
        print(f"🎯 NO SPLIT SCREENS will be created - all segments are too far apart (> {config.SPLIT_SCREEN_TIME_THRESHOLD}s)")
    
    return overlapping_groups

def create_split_screen_segment(overlap_group, split_screen_module):
    """Create a split screen video for overlapping segments."""
    segments = overlap_group['segments']
    start_time = overlap_group['start_time']
    end_time = overlap_group['end_time']
    duration = end_time - start_time
    
    # Get the earliest sequence number from the group for ordering
    sequence_number = min(seg.get('sequence_number', 0) for seg in segments)
    
    # Get unique celebrities in this group
    unique_celebrities = list(set(seg['celebrity'] for seg in segments))
    
    print(f"\n🎬 Processing group with {len(segments)} segments and {len(unique_celebrities)} unique celebrities")
    print(f"   Celebrities: {', '.join(unique_celebrities)}")
    print(f"   Sequence: {sequence_number:03d} (for timeline ordering)")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Timeframe: {start_time:.2f}-{end_time:.2f}s")
    
    # Handle different cases based on number of unique celebrities
    if len(unique_celebrities) == 1:
        print(f"   ⚠️  Only one celebrity ({unique_celebrities[0]}) - no split screen needed")
        return None
    elif len(unique_celebrities) == 2:
        # Perfect case - exactly 2 celebrities
        celebrity1, celebrity2 = unique_celebrities
        print(f"   ✨ Perfect match for split screen: {celebrity1} vs {celebrity2}")
    elif len(unique_celebrities) > 2:
        # Multiple celebrities - pick the two most prominent ones
        celebrity_durations = {}
        for celebrity in unique_celebrities:
            total_duration = sum(
                seg['media']['duration'] for seg in segments 
                if seg['celebrity'] == celebrity
            )
            celebrity_durations[celebrity] = total_duration
        
        # Sort by total duration and pick top 2
        sorted_celebrities = sorted(celebrity_durations.items(), key=lambda x: x[1], reverse=True)
        celebrity1, duration1 = sorted_celebrities[0]
        celebrity2, duration2 = sorted_celebrities[1]
        
        print(f"   🎯 Multiple celebrities detected - using top 2 by screen time:")
        print(f"      1. {celebrity1}: {duration1:.2f}s")
        print(f"      2. {celebrity2}: {duration2:.2f}s")
        
        if len(sorted_celebrities) > 2:
            print(f"      (Skipping: {', '.join([c[0] for c in sorted_celebrities[2:]])})")
    
    try:
        # Create temporary Split Screen instance with the detected celebrities
        temp_folder = Path(config.OUTPUT_FOLDER)
        temp_folder.mkdir(exist_ok=True)
        
        # Generate unique output filename with sequence number for timeline ordering
        celebrity1_safe = celebrity1.replace(' ', '_')
        celebrity2_safe = celebrity2.replace(' ', '_')
        output_filename = f"{sequence_number:03d}_split_screen_{celebrity1_safe}_{celebrity2_safe}_{start_time:.0f}s.mp4"
        output_path = temp_folder / output_filename
        
        # Temporarily modify the Split Screen module's keywords
        original_left = getattr(split_screen_module, 'LEFT_KEYWORD', None)
        original_right = getattr(split_screen_module, 'RIGHT_KEYWORD', None)
        original_duration = getattr(split_screen_module, 'CLIP_DURATION', None)
        
        # Set the celebrities for this split screen
        split_screen_module.LEFT_KEYWORD = celebrity1
        split_screen_module.RIGHT_KEYWORD = celebrity2
        split_screen_module.CLIP_DURATION = max(config.SPLIT_SCREEN_DURATION, duration)
        
        # Create the split screen effect
        # We'll call the main function but capture the output
        original_output_folder = split_screen_module.OUTPUT_FOLDER
        split_screen_module.OUTPUT_FOLDER = str(temp_folder)
        
        print(f"   🎬 Creating split screen with your Split Screen.py...")
        print(f"      Left: {celebrity1}")
        print(f"      Right: {celebrity2}")
        print(f"      Duration: {split_screen_module.CLIP_DURATION}s")
        print(f"      Timeline Output: {output_filename}")
        
        # Call the split screen creation function
        split_screen_module.create_split_screen_effect_ffmpeg()
        
        # Find the generated file (it might have a different name)
        generated_file = None
        
        for file in temp_folder.glob("split_screen_*.mp4"):
            if celebrity1_safe in file.name and celebrity2_safe in file.name:
                generated_file = file
                break
        
        if generated_file and generated_file.exists():
            # Rename to our desired filename if needed
            if generated_file.name != output_filename:
                final_output_path = temp_folder / output_filename
                generated_file.rename(final_output_path)
                output_path = final_output_path
            
            print(f"   ✓ Split screen created successfully!")
            print(f"   Output: {output_path}")
            
            # Create a new segment for the split screen
            split_screen_segment = {
                'start_time': start_time,
                'end_time': end_time,
                'type': 'celebrity_overlay',
                'celebrity': f"{celebrity1} & {celebrity2}",  # Combined celebrity
                'sequence_number': sequence_number,  # Maintain sequence ordering
                'media': {
                    'folder': str(temp_folder),
                    'type': 'video',
                    'duration': duration,
                    'file': output_filename
                },
                'effects_applied': ['split_screen'],
                'split_screen_celebrities': [celebrity1, celebrity2],
                'original_segments_count': len(segments)
            }
            
            # Restore original values
            if original_left is not None:
                split_screen_module.LEFT_KEYWORD = original_left
            if original_right is not None:
                split_screen_module.RIGHT_KEYWORD = original_right
            if original_duration is not None:
                split_screen_module.CLIP_DURATION = original_duration
            split_screen_module.OUTPUT_FOLDER = original_output_folder
            
            return split_screen_segment
            
        else:
            print(f"   ✗ Split screen generation failed - output file not found")
            # Restore original values even on failure
            if original_left is not None:
                split_screen_module.LEFT_KEYWORD = original_left
            if original_right is not None:
                split_screen_module.RIGHT_KEYWORD = original_right
            if original_duration is not None:
                split_screen_module.CLIP_DURATION = original_duration
            split_screen_module.OUTPUT_FOLDER = original_output_folder
            return None
            
    except Exception as e:
        print(f"   ✗ Failed to create split screen: {e}")
        # Restore original values on exception
        try:
            if original_left is not None:
                split_screen_module.LEFT_KEYWORD = original_left
            if original_right is not None:
                split_screen_module.RIGHT_KEYWORD = original_right
            if original_duration is not None:
                split_screen_module.CLIP_DURATION = original_duration
            split_screen_module.OUTPUT_FOLDER = original_output_folder
        except:
            pass
        return None

def resolve_conflicts_with_split_screen(segments, split_screen_module):
    """Resolve overlapping segments by creating split screen videos."""
    print(f"\n{'='*60}")
    print(f"CONFLICT RESOLUTION - Split Screen Processing")
    print(f"{'='*60}")
    
    # Check if split screen is enabled
    if not config.SPLIT_SCREEN_ENABLED:
        print("📱 Split Screen Effect: DISABLED (from Notion configuration)")
        print("⏭️  Skipping split screen processing - keeping original segments")
        return segments
    
    print("📱 Split Screen Effect: ENABLED (from Notion configuration)")
    
    # Detect overlapping segments
    overlapping_groups = detect_overlapping_segments(segments)
    
    if not overlapping_groups:
        print("✓ No overlapping segments found - no conflicts to resolve")
        return segments
    
    # Create new segments list
    new_segments = []
    processed_indices = set()
    
    # Process overlap groups
    for overlap_group in overlapping_groups:
        split_screen_segment = create_split_screen_segment(overlap_group, split_screen_module)
        
        if split_screen_segment:
            new_segments.append(split_screen_segment)
            processed_indices.update(overlap_group['indices'])
        else:
            # If split screen failed, keep original segments
            print(f"   ⚠️  Keeping original overlapping segments")
            for idx in overlap_group['indices']:
                new_segments.append(segments[idx])
            processed_indices.update(overlap_group['indices'])
    
    # Add non-overlapping segments
    for i, segment in enumerate(segments):
        if i not in processed_indices:
            new_segments.append(segment)
    
    # Sort by start time
    new_segments.sort(key=lambda x: x['start_time'])
    
    print(f"\n✓ Conflict resolution complete:")
    print(f"   • Original segments: {len(segments)}")
    print(f"   • Final segments: {len(new_segments)}")
    print(f"   • Split screens created: {len([s for s in new_segments if 'split_screen' in s.get('effects_applied', [])])}")
    
    return new_segments

def process_image_with_ken_burns(segment, ken_burns_exporter):
    """Process an image segment with Ken Burns effect."""
    media = segment['media']
    folder = Path(media['folder'])
    filename = media['file']
    duration = media['duration']
    celebrity = segment['celebrity']  # Get celebrity name for unique filenames
    sequence_number = segment.get('sequence_number', 0)  # Get sequence number for ordering
    
    input_path = folder / filename
    
    # Create a filename that maintains order but avoids special characters
    # Format: 001_Prince_Harry_clip.mp4
    def sanitize_for_filename(text):
        # Keep only letters, numbers, and spaces
        import re
        # First replace common special characters with spaces
        text = text.replace('&', ' and ')
        text = text.replace('-', ' ')
        text = text.replace('_', ' ')
        # Then remove any other special characters
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Convert spaces to underscores and trim
        text = text.strip().replace(' ', '_')
        return text
    
    celebrity_safe = sanitize_for_filename(celebrity)
    output_filename = f"{sequence_number:03d}_{celebrity_safe}_clip.mp4"
    
    # Render to configured temp folder
    temp_folder = Path(config.OUTPUT_FOLDER)
    temp_folder.mkdir(exist_ok=True)
    output_path = temp_folder / output_filename
    
    print(f"\n🖼️  Processing IMAGE: {filename}")
    print(f"   Celebrity: {celebrity}")
    print(f"   Sequence: {sequence_number:03d} (for timeline ordering)")
    print(f"   Duration: {duration} seconds")
    print(f"   🎨 KEN BURNS EFFECT: Constant zoom speed regardless of duration")
    print(f"   Timeline Output: {output_filename}")
    print(f"   Output location: {temp_folder}")
    
    try:
        # Use a fixed optimal zoom duration for consistent speed
        OPTIMAL_ZOOM_DURATION = 5.0  # Sweet spot for smooth zoom
        
        # Create Ken Burns exporter with fixed optimal duration
        exporter = ken_burns_exporter(
            output_width=config.OUTPUT_WIDTH,
            output_height=config.OUTPUT_HEIGHT,
            fps=config.OUTPUT_FPS,
            duration_seconds=int(OPTIMAL_ZOOM_DURATION)  # Always use optimal duration
        )
        
        # Use fixed zoom parameters for consistency
        exporter.max_zoom = 1.2  # Standard zoom range
        exporter.min_zoom = 1.0
        print(f"   📐 Using standard zoom range (1.0x → 1.2x)")
        
        # Use a simple temp filename that maintains sequence
        temp_output = temp_folder / f"{sequence_number:03d}_temp.mp4"
        
        # Convert paths to absolute and normalize them
        input_abs_path = input_path.resolve()
        temp_abs_path = temp_output.resolve()
        output_abs_path = output_path.resolve()
        
        print(f"   📁 Using absolute paths to avoid encoding issues:")
        print(f"      Input: {input_abs_path}")
        print(f"      Temp: {temp_abs_path}")
        print(f"      Output: {output_abs_path}")
        
        # Create Ken Burns effect
        exporter.export_video(str(input_abs_path), str(temp_abs_path))
        
        print(f"   ✓ Ken Burns effect created with optimal duration")
        print(f"   ✂️  Now trimming to exact duration: {duration:.3f} seconds")
        
        # Then use FFmpeg to trim to exact duration
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_abs_path),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            str(output_abs_path)
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        # Clean up temporary file
        try:
            if temp_output.exists():
                temp_output.unlink()
        except Exception as e:
            print(f"   ⚠️  Could not delete temp file: {e}")
        
        if result.returncode == 0:
            print(f"   ✓ Successfully trimmed to {duration:.3f}s")
            print(f"   Output: {output_path}")
            
            # Update the segment to point to the new video file
            segment['media']['file'] = output_filename
            segment['media']['folder'] = str(temp_folder)  # Update folder location
            segment['media']['type'] = 'video'  # Now it's a video
            segment['effects_applied'] = ['ken_burns']  # Track applied effects
            
            return True
            
        else:
            print(f"   ✗ Failed to trim video: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed to apply Ken Burns effect: {e}")
        print(f"   📁 Input path: {input_path}")
        print(f"   📁 Output path: {output_path}")
        return False

def process_video_with_composite(segment, composite_effect_class, video_index):
    """Process a video segment with optional Composite effect based on configuration."""
    media = segment['media']
    folder = Path(media['folder'])
    filename = media['file']
    duration = media['duration']
    celebrity = segment['celebrity']  # Get celebrity name for unique filenames
    sequence_number = segment.get('sequence_number', video_index)  # Get sequence number for ordering
    
    input_path = folder / filename
    
    # Check for invalid/corrupted files that should be skipped
    if filename.startswith('._'):
        print(f"\n⚠️  SKIPPING: {filename} (macOS metadata file - not a valid video)")
        return False
    
    if not input_path.exists():
        print(f"\n⚠️  SKIPPING: {filename} (file not found)")
        return False
    
    # Quick validation check for video file integrity
    try:
        # Quick ffprobe check to see if file is valid
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', str(input_path)
        ], capture_output=True, timeout=5)
        
        if result.returncode != 0:
            print(f"\n⚠️  SKIPPING: {filename} (corrupted or invalid video file)")
            return False
            
    except Exception as e:
        print(f"\n⚠️  SKIPPING: {filename} (validation failed: {e})")
        return False
    
    # Check if this video should get composite effect based on configuration
    should_apply_composite = (
        (video_index - config.COMPOSITE_START_INDEX) >= 0 and 
        (video_index - config.COMPOSITE_START_INDEX) % config.COMPOSITE_APPLY_INTERVAL == 0
    )
    
    # Render to configured temp folder
    temp_folder = Path(config.OUTPUT_FOLDER)
    temp_folder.mkdir(exist_ok=True)
    
    # Create UNIQUE filename with sequence number for perfect timeline ordering
    celebrity_safe = celebrity.replace(' ', '_').replace('&', 'and')
    
    if should_apply_composite:
        # Apply composite effect
        output_filename = f"{sequence_number:03d}_{celebrity_safe}_{Path(filename).stem}_composite.mp4"
        output_path = temp_folder / output_filename
        
        print(f"\n🎬 Processing VIDEO: {filename} (Index: {video_index})")
        print(f"   Celebrity: {celebrity}")
        print(f"   Sequence: {sequence_number:03d} (for timeline ordering)")
        print(f"   Duration: {duration} seconds")
        print(f"   🎨 COMPOSITE EFFECT: Scale + Background + Border overlay")
        print(f"   Config: Every {config.COMPOSITE_APPLY_INTERVAL} clips starting from {config.COMPOSITE_START_INDEX}")
        print(f"   Timeline Output: {output_filename}")
        print(f"   Output location: {temp_folder}")
        
        try:
            # Create composite effect instance with Notion background
            composite = composite_effect_class(
                output_width=config.OUTPUT_WIDTH,
                output_height=config.OUTPUT_HEIGHT,
                fps=config.OUTPUT_FPS,
                scale_factor=config.COMPOSITE_SCALE_FACTOR,
                notion_token="ntn_cC7520095381SElmcgTOADYsGnrABFn2ph1PrcaGSst2dv",  # Use Notion for background
                ffmpeg_config=FFMPEG_CONFIG  # Pass platform-specific FFmpeg config
            )
            
            # Apply composite effect using FFmpeg (much faster)
            try:
                composite.create_composite_ffmpeg(
                    main_video_path=str(input_path),
                    output_path=str(output_path),
                    max_duration=duration
                )
                method = f"FFmpeg ({FFMPEG_CONFIG['hwaccel'] or 'CPU'} accelerated)"
            except Exception as ffmpeg_error:
                print(f"   FFmpeg failed: {ffmpeg_error}")
                print(f"   Trying OpenCV fallback...")
                composite.create_composite(
                    main_video_path=str(input_path),
                    output_path=str(output_path),
                    max_duration=duration
                )
                method = "OpenCV (CPU optimized)"
            
            print(f"   ✓ Composite effect applied successfully!")
            print(f"   Method: {method}")
            print(f"   Output: {output_path}")
            
            # Update the segment to point to the new video file
            segment['media']['file'] = output_filename
            segment['media']['folder'] = str(temp_folder)
            segment['effects_applied'] = ['composite']  # Track applied effects
            
            return True
            
        except Exception as e:
            print(f"   ✗ Failed to apply composite effect: {e}")
            print(f"   📝 Suggestion: Check if {filename} is a valid video file")
            
            # Instead of failing completely, copy the original as fallback
            try:
                fallback_filename = f"{sequence_number:03d}_{celebrity_safe}_{Path(filename).stem}_original.mp4"
                fallback_path = temp_folder / fallback_filename
                
                print(f"   🔄 Composite failed, trimming original to exact duration as fallback")
                print(f"   ✂️  Trimming to: {duration:.3f} seconds")
                
                # Use FFmpeg to trim the original video to exact duration with platform-specific settings
                ffmpeg_cmd = [
                    'ffmpeg', '-y'
                ]
                
                # Add hardware acceleration if available
                if FFMPEG_CONFIG['hwaccel']:
                    ffmpeg_cmd.extend(['-hwaccel', FFMPEG_CONFIG['hwaccel']])
                
                ffmpeg_cmd.extend([
                    '-i', str(input_path),
                    '-t', str(duration),
                    '-c:v', FFMPEG_CONFIG['video_codec'],
                    '-c:a', 'aac',
                    '-preset', FFMPEG_CONFIG['preset'],
                    '-pix_fmt', FFMPEG_CONFIG['pixel_format'],
                    str(fallback_path)
                ])
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   ✓ Fallback: Original trimmed to {duration:.3f}s")
                    segment['media']['file'] = fallback_filename
                    segment['media']['folder'] = str(temp_folder)
                    segment['effects_applied'] = ['trimmed']
                    return True
                else:
                    # If trimming fails, do simple copy as last resort
                    print(f"   ⚠️  Trimming failed, doing simple copy (may cause timeline drift)")
                    import shutil
                    shutil.copy2(str(input_path), str(fallback_path))
                    segment['media']['file'] = fallback_filename
                    segment['media']['folder'] = str(temp_folder)
                    segment['effects_applied'] = []
                    return True
                
            except Exception as copy_error:
                print(f"   ✗ Fallback copy also failed: {copy_error}")
                return False
    
    else:
        # Skip composite effect - just copy/reference original with unique name
        print(f"\n📹 Processing VIDEO: {filename} (Index: {video_index})")
        print(f"   Celebrity: {celebrity}")
        print(f"   Sequence: {sequence_number:03d} (for timeline ordering)")
        print(f"   Duration: {duration} seconds")
        print(f"   🚫 NO COMPOSITE EFFECT (Config: Every {config.COMPOSITE_APPLY_INTERVAL} clips)")
        print(f"   Reason: Index {video_index} not in composite schedule")
        
        # Create unique filename including celebrity name
        output_filename = f"{sequence_number:03d}_{celebrity_safe}_{Path(filename).stem}_original.mp4"
        output_path = temp_folder / output_filename
        
        try:
            # Trim video to exact duration using FFmpeg with platform-specific settings
            print(f"   ✂️  Trimming to exact duration: {duration:.3f} seconds")
            print(f"   📐 This prevents timeline drift in your video editor")
            
            # Use FFmpeg to trim video to exact duration
            ffmpeg_cmd = [
                'ffmpeg', '-y'  # -y to overwrite output file
            ]
            
            # Add hardware acceleration if available
            if FFMPEG_CONFIG['hwaccel']:
                ffmpeg_cmd.extend(['-hwaccel', FFMPEG_CONFIG['hwaccel']])
            
            # Get video start time using ffprobe
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(input_path)
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            video_info = json.loads(probe_result.stdout)
            
            # Get start time and input fps
            start_time = float(video_info['format'].get('start_time', '0'))
            
            # Add input with precise seeking
            ffmpeg_cmd.extend([
                '-i', str(input_path),  # Input file
                '-ss', str(start_time),  # Seek to start_time
                '-t', str(duration),     # Trim to exact duration
                '-c:v', FFMPEG_CONFIG['video_codec'],  # Platform-specific video codec
                '-c:a', 'aac',           # Audio codec  
                '-preset', FFMPEG_CONFIG['preset'],  # Platform-specific preset
                '-pix_fmt', FFMPEG_CONFIG['pixel_format'],  # Platform-specific pixel format
                '-vsync', 'cfr',         # Force constant frame rate
                '-r', str(config.OUTPUT_FPS),  # Force output fps
                '-fflags', '+genpts',    # Generate presentation timestamps
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                str(output_path)         # Output file
            ])
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✓ Video trimmed successfully to {duration:.3f}s")
                print(f"   Timeline Output: {output_filename}")
                print(f"   Output: {output_path}")
                
                # Update the segment to point to the trimmed file
                segment['media']['file'] = output_filename
                segment['media']['folder'] = str(temp_folder)
                segment['effects_applied'] = ['trimmed']  # Track that it was trimmed
                
                return True
            else:
                print(f"   ✗ FFmpeg trimming failed: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")
                
        except Exception as e:
            print(f"   ✗ Failed to trim video with FFmpeg: {e}")
            print(f"   🔄 Falling back to simple copy (will cause timeline drift)")
            
            try:
                # Fallback: Simple copy (will cause timeline issues but at least works)
                import shutil
                shutil.copy2(str(input_path), str(output_path))
                
                print(f"   ⚠️  WARNING: Copied full-length file - may cause timeline drift")
                print(f"   Timeline Output: {output_filename}")
                print(f"   Output: {output_path}")
                
                # Update the segment to point to the copied file
                segment['media']['file'] = output_filename
                segment['media']['folder'] = str(temp_folder)
                segment['effects_applied'] = []  # No effects applied
                
                return True
                
            except Exception as copy_error:
                print(f"   ✗ Fallback copy also failed: {copy_error}")
                return False

def create_human_readable_effects_summary(render_plan):
    """Create a human-readable text summary of the effects applied to segments."""
    output_file = json_files_dir / "5. Render Plan V2 - Effects Applied.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("EFFECTS PROCESSING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Video Info
            video_info = render_plan.get('video_info', {})
            f.write(f"Video Title: {video_info.get('title', 'Unknown')}\n")
            f.write(f"Effects Applied: {video_info.get('generated_at', 'Unknown')}\n")
            f.write(f"Total Segments: {len(render_plan.get('segments', []))}\n")
            f.write(f"Output Resolution: {config.OUTPUT_WIDTH}x{config.OUTPUT_HEIGHT}\n")
            f.write(f"Output FPS: {config.OUTPUT_FPS}\n")
            f.write(f"Composite Interval: Every {config.COMPOSITE_APPLY_INTERVAL} clips\n\n")
            
            # Effects Configuration Summary
            f.write("-" * 80 + "\n")
            f.write("EFFECTS CONFIGURATION\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"🎬 Composite Effect:\n")
            f.write(f"   • Apply Interval: Every {config.COMPOSITE_APPLY_INTERVAL} video clips\n")
            f.write(f"   • Scale Factor: {config.COMPOSITE_SCALE_FACTOR} ({int(config.COMPOSITE_SCALE_FACTOR*100)}%)\n")
            f.write(f"   • Background: Notion-configured\n")
            f.write(f"   • Border Overlay: Assets/Background/Box.png\n")
            f.write(f"   • Duration: Trimmed to exact render plan timing\n\n")
            
            f.write(f"��️  Ken Burns Effect:\n")
            f.write(f"   • Apply to: All image segments\n")
            f.write(f"   • Effect: Random zoom in/out with smooth motion\n")
            f.write(f"   • Duration: Matches segment duration exactly\n\n")
            
            f.write(f"📱 Split Screen Effect:\n")
            f.write(f"   • Status: {'ENABLED' if config.SPLIT_SCREEN_ENABLED else 'DISABLED'} (from Notion)\n")
            f.write(f"   • Trigger: Overlapping/simultaneous celebrity detections\n")
            f.write(f"   • Time Threshold: {config.SPLIT_SCREEN_TIME_THRESHOLD} seconds\n")
            f.write(f"   • Duration: {config.SPLIT_SCREEN_DURATION} seconds minimum\n")
            
            f.write(f"✂️  Precision Trimming:\n")
            f.write(f"   • ALL clips trimmed to exact render plan durations\n")
            f.write(f"   • Prevents timeline drift in video editors\n")
            f.write(f"   • Perfect synchronization with voiceover timing\n")
            f.write(f"   • Uses FFmpeg for frame-accurate precision\n\n")
            
            # Timeline Overview with Effects
            f.write("-" * 80 + "\n")
            f.write("TIMELINE OVERVIEW WITH EFFECTS\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort segments by start time
            segments = render_plan.get('segments', [])
            sorted_segments = sorted(segments, key=lambda x: x['start_time'])
            
            current_celebrity = ""
            segment_count = 0
            
            for i, segment in enumerate(sorted_segments):
                start_time = segment['start_time']
                end_time = segment['end_time']
                duration = segment['media']['duration']
                celebrity = segment['celebrity']
                media_type = segment['media']['type']
                filename = segment['media']['file']
                effects_applied = segment.get('effects_applied', [])
                
                # Format time as MM:SS
                start_min, start_sec = divmod(int(start_time), 60)
                end_min, end_sec = divmod(int(end_time), 60)
                start_ms = int((start_time % 1) * 100)
                end_ms = int((end_time % 1) * 100)
                
                # Check if we're starting a new celebrity section
                if celebrity != current_celebrity:
                    if current_celebrity:  # Not the first celebrity
                        f.write("\n")
                    f.write(f"🎬 {celebrity.upper()}\n")
                    f.write("-" * 40 + "\n")
                    current_celebrity = celebrity
                    segment_count = 0
                
                segment_count += 1
                
                # Write segment info with effects
                if media_type == "image":
                    media_icon = "🖼️"
                    media_desc = " (Ken Burns)"
                elif "split_screen" in effects_applied:
                    media_icon = "📱"
                    media_desc = " (Split Screen)"
                elif "composite" in effects_applied:
                    media_icon = "🎬"
                    media_desc = " (Composite)"
                else:
                    media_icon = "🎥"
                    media_desc = " (Original)"
                
                f.write(f"  {segment_count:2d}. {media_icon} {start_min:02d}:{start_sec:02d}.{start_ms:02d} → {end_min:02d}:{end_sec:02d}.{end_ms:02d} ")
                f.write(f"({duration:.2f}s){media_desc} - {filename}\n")
                
                # Show detailed effects info if any
                if effects_applied:
                    f.write(f"       Effects: {', '.join(effects_applied)}\n")
                
                # Show special info for split screens
                if "split_screen" in effects_applied:
                    split_celebrities = segment.get('split_screen_celebrities', [])
                    if split_celebrities:
                        f.write(f"       Split: {' vs '.join(split_celebrities)}\n")
            
            # Effects Statistics
            f.write("\n" + "-" * 80 + "\n")
            f.write("EFFECTS STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            # Count different types of effects
            ken_burns_count = 0
            composite_count = 0
            split_screen_count = 0
            original_count = 0
            total_duration = 0
            
            celebrity_stats = {}
            
            for segment in sorted_segments:
                celebrity = segment['celebrity']
                duration = segment['media']['duration']
                effects_applied = segment.get('effects_applied', [])
                
                if celebrity not in celebrity_stats:
                    celebrity_stats[celebrity] = {
                        'total_duration': 0,
                        'ken_burns': 0,
                        'composite': 0,
                        'split_screen': 0,
                        'original': 0,
                        'total_clips': 0
                    }
                
                celebrity_stats[celebrity]['total_duration'] += duration
                celebrity_stats[celebrity]['total_clips'] += 1
                total_duration += duration
                
                if 'ken_burns' in effects_applied:
                    ken_burns_count += 1
                    celebrity_stats[celebrity]['ken_burns'] += 1
                elif 'composite' in effects_applied:
                    composite_count += 1
                    celebrity_stats[celebrity]['composite'] += 1
                elif 'split_screen' in effects_applied:
                    split_screen_count += 1
                    celebrity_stats[celebrity]['split_screen'] += 1
                else:
                    original_count += 1
                    celebrity_stats[celebrity]['original'] += 1
            
            # Overall effects summary
            f.write(f"Overall Effects Applied:\n")
            f.write(f"  🖼️  Ken Burns (Images): {ken_burns_count} segments\n")
            f.write(f"  🎬 Composite (Videos): {composite_count} segments\n")
            f.write(f"  📱 Split Screen (Conflicts): {split_screen_count} segments\n")
            f.write(f"  🎥 Original (No effects): {original_count} segments\n")
            f.write(f"  📊 Total Segments: {len(sorted_segments)}\n\n")
            
            # Celebrity breakdown with effects
            f.write(f"Effects by Celebrity:\n")
            for celebrity, stats in celebrity_stats.items():
                percentage = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
                f.write(f"\n{celebrity}:\n")
                f.write(f"  • Total Screen Time: {stats['total_duration']:.2f} seconds ({percentage:.1f}%)\n")
                f.write(f"  • Total Clips: {stats['total_clips']}\n")
                f.write(f"  • Ken Burns: {stats['ken_burns']} clips\n")
                f.write(f"  • Composite: {stats['composite']} clips\n")
                f.write(f"  • Split Screen: {stats['split_screen']} clips\n")
                f.write(f"  • Original: {stats['original']} clips\n")
            
            # Processing Summary
            f.write(f"\n" + "-" * 80 + "\n")
            f.write("PROCESSING SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            effects_efficiency = ((ken_burns_count + composite_count + split_screen_count) / len(sorted_segments) * 100) if sorted_segments else 0
            
            f.write(f"📈 Processing Efficiency:\n")
            f.write(f"  • Effects Applied: {ken_burns_count + composite_count + split_screen_count}/{len(sorted_segments)} segments ({effects_efficiency:.1f}%)\n")
            f.write(f"  • Conflicts Resolved: {split_screen_count} overlapping segments\n")
            f.write(f"  • Images Enhanced: {ken_burns_count} with Ken Burns effect\n")
            f.write(f"  • Videos Processed: {composite_count} with composite effects\n\n")
            
            f.write(f"💾 Output Files:\n")
            f.write(f"  • Location: {config.OUTPUT_FOLDER}\n")
            f.write(f"  • Naming: [original]_[effect].mp4\n")
            f.write(f"  • Quality: {config.OUTPUT_WIDTH}x{config.OUTPUT_HEIGHT} @ {config.OUTPUT_FPS} FPS\n\n")
            
            f.write(f"🎯 Next Steps:\n")
            f.write(f"  1. Review processed segments in: {config.OUTPUT_FOLDER}\n")
            f.write(f"  2. Run Final Assembly script to combine all segments\n")
            f.write(f"  3. Add voice-over audio to complete the video\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Effects Processing Summary\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Human-readable effects summary created: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating human-readable effects summary: {str(e)}")
        return False

def copy_voiceover_to_output_folder():
    """Copy the voiceover file from Celebrity Voice Overs to the output folder."""
    # Platform-specific paths
    if SYSTEM == 'darwin':  # macOS
        voiceover_source_dir = Path("/Users/superman/Desktop/Celebrity Voice Overs")
        output_dir = Path("/Users/superman/Desktop/Clips Assembled")
    else:  # Windows
        voiceover_source_dir = Path(r"E:\Celebrity Voice Overs")
        output_dir = Path(config.OUTPUT_FOLDER)
    
    print(f"\n🎵 COPYING VOICEOVER FILE")
    print(f"{'='*60}")
    
    # Load selected video info to get the title
    video_info = load_selected_video_info()
    if not video_info or 'title' not in video_info:
        print("❌ Could not load video title from Selected Video.json")
        return False
    
    video_title = video_info['title']
    print(f"🎬 Video Title: {video_title}")
    print(f"📁 Source Directory: {voiceover_source_dir}")
    print(f"📁 Destination Directory: {output_dir}")
    
    # Check if source directory exists
    if not voiceover_source_dir.exists():
        print(f"❌ Source directory does not exist: {voiceover_source_dir}")
        return False
    
    # Common audio file extensions
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
    
    # Look for audio file with matching title
    found_file = None
    
    print(f"\n🔍 Searching for voiceover file...")
    
    for ext in audio_extensions:
        potential_file = voiceover_source_dir / f"{video_title}{ext}"
        print(f"   Checking: {video_title}{ext}")
        
        if potential_file.exists():
            found_file = potential_file
            print(f"   ✅ FOUND: {potential_file.name}")
            break
        else:
            print(f"   ❌ Not found")
    
    if not found_file:
        print(f"\n❌ No voiceover file found for title: {video_title}")
        print(f"💡 Checked these extensions: {', '.join(audio_extensions)}")
        print(f"📂 Available files in {voiceover_source_dir}:")
        
        try:
            available_files = list(voiceover_source_dir.glob("*"))[:10]  # Show first 10 files
            for file in available_files:
                if file.is_file():
                    print(f"   • {file.name}")
            if len(list(voiceover_source_dir.glob("*"))) > 10:
                print(f"   ... and {len(list(voiceover_source_dir.glob('*'))) - 10} more files")
        except Exception as e:
            print(f"   Error listing files: {e}")
        
        return False
    
    # Copy the voiceover file to output folder
    try:
        destination_file = output_dir / found_file.name
        
        print(f"\n📋 Copying voiceover file...")
        print(f"   From: {found_file}")
        print(f"   To: {destination_file}")
        
        import shutil
        shutil.copy2(found_file, destination_file)
        
        print(f"✅ Voiceover copied successfully!")
        print(f"   📄 File: {found_file.name}")
        print(f"   📁 Location: {output_dir}")
        
        # Get file size for confirmation
        file_size = destination_file.stat().st_size
        if file_size > 1024 * 1024:  # > 1MB
            size_mb = file_size / (1024 * 1024)
            print(f"   📊 Size: {size_mb:.1f} MB")
        else:
            size_kb = file_size / 1024
            print(f"   📊 Size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to copy voiceover file: {e}")
        return False

def create_drag_drop_workflow_summary(render_plan):
    """Create a summary showing the drag-and-drop workflow for manual timeline assembly."""
    output_file = json_files_dir / "6. Drag and Drop Workflow Guide.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("DRAG & DROP TIMELINE WORKFLOW GUIDE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("🎯 PERFECT TIMELINE ASSEMBLY WORKFLOW\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("Your video clips have been processed and saved to:\n")
            f.write(f"📁 {config.OUTPUT_FOLDER}\n\n")
            
            f.write("✨ KEY FEATURE: All files are named with sequence numbers (001, 002, 003...)\n")
            f.write("   so they will sort in PERFECT TIMELINE ORDER when you view them!\n\n")
            
            f.write("🎬 HOW TO ASSEMBLE YOUR VIDEO:\n")
            f.write("1. Open your video editor (Premiere Pro, DaVinci Resolve, etc.)\n")
            f.write("2. Create a new project with these settings:\n")
            f.write(f"   • Resolution: {config.OUTPUT_WIDTH}x{config.OUTPUT_HEIGHT}\n")
            f.write(f"   • Frame Rate: {config.OUTPUT_FPS} FPS\n")
            f.write("3. Navigate to the output folder in File Explorer\n")
            f.write("4. Sort files by name (they'll be in perfect timeline order)\n")
            f.write("5. Select ALL processed clips (Ctrl+A)\n")
            f.write("6. Drag them to your timeline - they'll align PERFECTLY!\n")
            f.write("7. Drag the voiceover audio file to an audio track\n")
            f.write("8. Export your final video\n\n")
            
            # Show the exact file ordering
            f.write("📋 EXACT FILE ORDER FOR YOUR TIMELINE:\n")
            f.write("-" * 50 + "\n")
            
            segments = render_plan.get('segments', [])
            sorted_segments = sorted(segments, key=lambda x: x.get('sequence_number', 0))
            
            total_duration = 0
            
            for segment in sorted_segments:
                seq_num = segment.get('sequence_number', 0)
                celebrity = segment.get('celebrity', 'Unknown')
                filename = segment.get('media', {}).get('file', 'unknown.mp4')
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                duration = segment.get('media', {}).get('duration', 0)
                effects = segment.get('effects_applied', [])
                
                total_duration += duration
                
                # Format timeline position
                start_min, start_sec = divmod(int(start_time), 60)
                start_ms = int((start_time % 1) * 100)
                
                effect_icon = "🎥"
                effect_desc = ""
                if 'split_screen' in effects:
                    effect_icon = "📱"
                    effect_desc = " (Split Screen)"
                elif 'ken_burns' in effects:
                    effect_icon = "🖼️"
                    effect_desc = " (Ken Burns)"
                elif 'composite' in effects:
                    effect_icon = "🎬"
                    effect_desc = " (Composite)"
                else:
                    effect_desc = " (Original)"
                
                f.write(f"{seq_num:03d}. {effect_icon} [{start_min:02d}:{start_sec:02d}.{start_ms:02d}] ")
                f.write(f"{celebrity}{effect_desc} ({duration:.2f}s)\n")
                f.write(f"     📄 File: {filename}\n")
            
            # Summary stats
            f.write(f"\n📊 SUMMARY:\n")
            f.write(f"• Total Clips: {len(sorted_segments)}\n")
            f.write(f"• Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)\n")
            f.write(f"• Output Location: {config.OUTPUT_FOLDER}\n")
            f.write(f"• File Naming: 001_Celebrity_filename_effect.mp4\n\n")
            
            f.write("⚡ ADVANTAGES OF THIS WORKFLOW:\n")
            f.write("• Perfect timing alignment (no manual adjustments needed)\n")
            f.write("• Full creative control in your preferred editor\n")
            f.write("• Easy to make cuts, transitions, and color corrections\n")
            f.write("• Can easily swap out clips or reorder if needed\n")
            f.write("• All effects pre-applied and optimized\n")
            f.write("• Ready for professional finishing touches\n\n")
            
            f.write("🎵 ADD YOUR VOICEOVER:\n")
            f.write("• Import your voiceover audio file\n")
            f.write("• Place it on a separate audio track\n")
            f.write("• The video clips are already perfectly timed to your script\n")
            f.write("• Fine-tune audio levels and sync as needed\n\n")
            
            f.write("💡 PRO TIPS:\n")
            f.write("• Keep the original render plan JSON for reference\n")
            f.write("• Use crossfade transitions between clips for smoothness\n")
            f.write("• Add background music on a third audio track\n")
            f.write("• Color grade the final timeline for consistency\n")
            f.write("• Export at high quality for final distribution\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Ready for Professional Video Assembly! 🎬✨\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ Drag & Drop workflow guide created: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating workflow guide: {str(e)}")
        return False

def main():
    """Main function to process the render plan and apply effects."""
    print("Effects Processor")
    print("================")
    print("Processing Render Plan V2.json with Conflict Resolution and Effects")
    print()
    
    # Load render plan
    render_plan = load_render_plan()
    if not render_plan:
        print("Failed to load render plan. Exiting.")
        return
    
    # Get segments
    segments = render_plan.get('segments', [])
    print(f"\nLoaded {len(segments)} segments from render plan")
    
    # Import required modules
    print("\nImporting effect modules...")
    
    # Import Split Screen effect
    split_screen_module = import_module_from_path(
        "split_screen_module",
        other_files_dir / "Split Screen.py"
    )
    if not split_screen_module:
        print("Failed to import Split Screen module. Exiting.")
        return
    
    # Import Ken Burns effect
    image_zoom_module = import_module_from_path(
        "image_zoom_module", 
        other_files_dir / "Image Zoom.py"
    )
    if not image_zoom_module:
        print("Failed to import Image Zoom module. Exiting.")
        return
    
    # Import Composite effect
    clip_composite_module = import_module_from_path(
        "clip_composite_module",
        other_files_dir / "Clip Composite.py"
    )
    if not clip_composite_module:
        print("Failed to import Clip Composite module. Exiting.")
        return
    
    print("✓ All effect modules imported successfully")
    
    # Get the classes we need
    KenBurnsVideoExporter = image_zoom_module.KenBurnsVideoExporter
    ClipCompositeEffect = clip_composite_module.ClipCompositeEffect
    
    # PHASE 1: Resolve conflicts with split screens
    segments = resolve_conflicts_with_split_screen(segments, split_screen_module)
    
    # Update render plan with conflict-resolved segments
    render_plan['segments'] = segments
    
    # PHASE 2: Apply visual effects to remaining segments
    print(f"\n{'='*60}")
    print(f"VISUAL EFFECTS PROCESSING")
    print(f"{'='*60}")
    print(f"Processing {len(segments)} segments with Ken Burns and Composite effects...")
    print(f"💡 Files will be named with sequence numbers for easy drag-and-drop timeline workflow!")
    
    # Add sequence numbers to segments for timeline ordering
    for i, segment in enumerate(segments):
        segment['sequence_number'] = i + 1  # 1-based indexing for clarity
    
    processed_images = 0
    processed_videos = 0
    failed_segments = 0
    failed_segment_details = []  # Track which segments failed
    
    for i, segment in enumerate(segments, 1):
        print(f"\n{'='*60}")
        print(f"SEGMENT {i}/{len(segments)}: {segment.get('celebrity', 'Unknown')}")
        print(f"{'='*60}")
        
        # Skip segments that already have split screen effect
        if 'split_screen' in segment.get('effects_applied', []):
            print(f"   ⏭️  SKIPPING: Split screen segment (already processed)")
            continue
        
        media = segment.get('media', {})
        media_type = media.get('type', 'unknown')
        
        if media_type == 'image':
            # Apply Ken Burns effect to image
            success = process_image_with_ken_burns(segment, KenBurnsVideoExporter)
            if success:
                processed_images += 1
            else:
                failed_segments += 1
                failed_segment_details.append(f"Segment {i}: {segment.get('celebrity', 'Unknown')} - {media.get('file', 'Unknown file')} (Ken Burns failed)")
                
        elif media_type == 'video':
            # Apply Composite effect to video
            success = process_video_with_composite(segment, ClipCompositeEffect, i)
            if success:
                processed_videos += 1
            else:
                failed_segments += 1
                failed_segment_details.append(f"Segment {i}: {segment.get('celebrity', 'Unknown')} - {media.get('file', 'Unknown file')} (Video processing failed)")
                
        else:
            print(f"\n⚠️  SKIPPING: Unknown media type '{media_type}' in segment {i}")
            failed_segments += 1
            failed_segment_details.append(f"Segment {i}: {segment.get('celebrity', 'Unknown')} - {media.get('file', 'Unknown file')} (Unknown media type: {media_type})")
    
    # Save updated render plan
    try:
        updated_plan_path = json_files_dir / "Render Plan V2 - Effects Applied.json"
        with open(updated_plan_path, 'w', encoding='utf-8') as f:
            json.dump(render_plan, f, indent=4)
        print(f"\n✓ Updated render plan saved: {updated_plan_path.name}")
    except Exception as e:
        print(f"\n✗ Failed to save updated render plan: {e}")
    
    # Summary
    split_screen_count = len([s for s in segments if 'split_screen' in s.get('effects_applied', [])])
    
    print(f"\n{'='*60}")
    print(f"EFFECTS PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"   • Split screens created: {split_screen_count}")
    print(f"   • Images processed (Ken Burns): {processed_images}")
    print(f"   • Videos processed (Composite): {processed_videos}")
    print(f"   • Failed segments: {failed_segments}")
    print(f"   • Total final segments: {len(segments)}")
    
    if failed_segments == 0:
        print(f"\n🎉 All effects applied successfully! ✨")
    else:
        print(f"\n⚠️  {failed_segments} segments failed processing.")
        print(f"\n📋 FAILED SEGMENTS DETAILS:")
        for detail in failed_segment_details:
            print(f"   • {detail}")
        
        print(f"\n💡 COMMON SOLUTIONS:")
        print(f"   • Check if all video files exist and aren't corrupted")
        if SYSTEM == 'windows':
            print(f"   • Ensure FFmpeg is properly installed with CUDA support")
            print(f"   • Update NVIDIA drivers and CUDA toolkit")
        elif SYSTEM == 'darwin':
            print(f"   • Ensure FFmpeg is installed with VideoToolbox support")
            print(f"   • Run 'brew upgrade ffmpeg' to get the latest version")
        else:
            print(f"   • Ensure FFmpeg is properly installed")
            print(f"   • Run 'sudo apt update && sudo apt upgrade ffmpeg' to update")
        print(f"   • Verify file formats are supported (MP4, MOV, AVI, etc.)")
        print(f"   • Check file permissions and disk space")
    
    print(f"\nNext step: Use the updated render plan for final video assembly.")

    # Create human-readable effects summary
    create_human_readable_effects_summary(render_plan)

    # Create drag-and-drop workflow summary
    create_drag_drop_workflow_summary(render_plan)
    
    # Update Notion checkbox to mark video as edited
    video_info = render_plan.get('video_info', {})
    video_title = video_info.get('title')
    if video_title:
        print(f"\n📝 Marking video as edited in Notion...")
        update_notion_checkbox(video_title)
    else:
        print(f"\n⚠️  No video title found - updating Notion without search filter")
        update_notion_checkbox()
    
    # Copy voiceover file to output folder
    copy_voiceover_to_output_folder()

if __name__ == "__main__":
    main()

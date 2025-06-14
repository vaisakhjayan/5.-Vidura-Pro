import json
import sys
import os
from pathlib import Path
import subprocess
import unicodedata
import re
from datetime import datetime
import time

# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

def log(message, level="info", newline=True):
    """Print a nicely formatted log message with timestamp and color, all italic."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if level == "info":
        prefix = f"{Colors.BLUE}ℹ{Colors.RESET}"
        color = Colors.RESET
    elif level == "success":
        prefix = f"{Colors.GREEN}✓{Colors.RESET}"
        color = Colors.GREEN
    elif level == "warn":
        prefix = f"{Colors.YELLOW}⚠{Colors.RESET}"
        color = Colors.YELLOW
    elif level == "error":
        prefix = f"{Colors.RED}✗{Colors.RESET}"
        color = Colors.RED
    elif level == "wait":
        prefix = f"{Colors.CYAN}◔{Colors.RESET}"
        color = Colors.CYAN
    elif level == "header":
        prefix = f"{Colors.MAGENTA}▶{Colors.RESET}"
        color = Colors.MAGENTA + Colors.BOLD
    elif level == "dim":
        prefix = f"{Colors.CYAN}◔{Colors.RESET}"
        color = Colors.DIM
    else:
        prefix = " "
        color = Colors.RESET
    
    log_msg = f"{Colors.ITALIC}{Colors.DIM}[{timestamp}]{Colors.RESET} {prefix} {color}{message}{Colors.RESET}{Colors.RESET}\033[0m"
    
    if newline:
        print(log_msg)
    else:
        print(log_msg, end="", flush=True)

# Add the JSON Files directory path
current_dir = Path(__file__).parent
json_files_dir = current_dir.parent / "JSON Files"

# =============================================================================
# FINAL ASSEMBLY CONFIGURATION
# =============================================================================
class AssemblyConfig:
    """Configuration settings for final video assembly."""
    
    # Input Settings
    EFFECTS_APPLIED_JSON = "Render Plan V2 - Effects Applied.json"
    
    # Output Settings
    OUTPUT_FOLDER = r"E:\Temp"
    FINAL_VIDEO_NAME = "final_video.mp4"
    
    # FFmpeg Settings
    USE_GPU_ENCODING = True        # Use h264_nvenc if available
    OUTPUT_QUALITY = "fast"        # fast, medium, slow
    COPY_STREAMS = True            # Use stream copy since segments are pre-processed

config = AssemblyConfig()

def load_detections():
    """Load the detections JSON file to get video title."""
    detections_path = json_files_dir / "Detections.json"
    
    try:
        with open(detections_path, 'r') as f:
            detections = json.load(f)
        print(f"✓ Loaded detections: {detections_path}")
        return detections
    except FileNotFoundError:
        print(f"Warning: Detections file not found at {detections_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in detections file: {e}")
        return None

def load_transcriptions_config():
    """Load the transcriptions JSON file to get dry run mode configuration."""
    transcriptions_path = json_files_dir / "2. Transcriptions.json"
    
    try:
        with open(transcriptions_path, 'r', encoding='utf-8') as f:
            transcriptions = json.load(f)
        print(f"✓ Loaded transcriptions config: {transcriptions_path}")
        
        # Extract dry run mode information from export metadata
        export_metadata = transcriptions.get('export_metadata', {})
        dry_run_config = export_metadata.get('dry_run_mode', {})
        
        dry_run_enabled = dry_run_config.get('enabled', False)
        dry_run_duration = dry_run_config.get('duration_seconds', None)
        dry_run_status = dry_run_config.get('status', 'Unknown')
        
        print(f"📡 Dry Run Configuration:")
        print(f"   Enabled: {dry_run_enabled}")
        print(f"   Duration: {dry_run_duration} seconds" if dry_run_duration else "   Duration: Full audio")
        print(f"   Status: {dry_run_status}")
        
        return {
            'enabled': dry_run_enabled,
            'duration_seconds': dry_run_duration,
            'status': dry_run_status
        }
        
    except FileNotFoundError:
        print(f"Warning: Transcriptions file not found at {transcriptions_path}")
        print(f"Assuming full audio mode (no dry run)")
        return {'enabled': False, 'duration_seconds': None, 'status': 'Full audio mode'}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in transcriptions file: {e}")
        return {'enabled': False, 'duration_seconds': None, 'status': 'Error reading config'}
    except Exception as e:
        print(f"Error loading transcriptions config: {e}")
        return {'enabled': False, 'duration_seconds': None, 'status': 'Error loading config'}

def normalize_text(text):
    """Normalize text for better matching by handling encoding and special characters."""
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Handle common encoding issues
    text = text.replace('Ã©', 'é')  # Fix Beyoncé encoding issue
    text = text.replace('Ã¡', 'á')
    text = text.replace('Ã³', 'ó')
    text = text.replace('Ã­', 'í')
    text = text.replace('Ã±', 'ñ')
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-\.\(\)]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def find_audio_file(title):
    """Find audio file in Celebrity Voice Overs folder based on title."""
    voice_overs_folder = Path(r"E:\Celebrity Voice Overs")
    
    if not voice_overs_folder.exists():
        print(f"Warning: Celebrity Voice Overs folder not found: {voice_overs_folder}")
        return None
    
    # Common audio extensions
    audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
    
    print(f"Searching for audio file with title: '{title}'")
    
    # Normalize the search title
    normalized_title = normalize_text(title)
    print(f"Normalized search title: '{normalized_title}'")
    
    # Try exact match first
    for ext in audio_extensions:
        audio_path = voice_overs_folder / f"{title}{ext}"
        if audio_path.exists():
            print(f"✓ Found exact match: {audio_path.name}")
            return audio_path
    
    # Try normalized exact match
    for ext in audio_extensions:
        audio_path = voice_overs_folder / f"{normalized_title}{ext}"
        if audio_path.exists():
            print(f"✓ Found normalized exact match: {audio_path.name}")
            return audio_path
    
    # Try case-insensitive search with normalization
    try:
        best_match = None
        best_score = 0
        
        for file_path in voice_overs_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                file_stem = normalize_text(file_path.stem)
                
                # Exact normalized match
                if file_stem.lower() == normalized_title.lower():
                    print(f"✓ Found case-insensitive normalized match: {file_path.name}")
                    return file_path
                
                # Partial match scoring (for very similar titles)
                if len(normalized_title) > 10:  # Only for reasonably long titles
                    # Count matching words
                    title_words = set(normalized_title.lower().split())
                    file_words = set(file_stem.lower().split())
                    common_words = title_words.intersection(file_words)
                    
                    if len(common_words) >= 3:  # At least 3 matching words
                        score = len(common_words) / max(len(title_words), len(file_words))
                        if score > best_score and score > 0.6:  # At least 60% match
                            best_match = file_path
                            best_score = score
        
        if best_match:
            print(f"✓ Found partial match ({best_score:.1%} similarity): {best_match.name}")
            return best_match
            
    except Exception as e:
        print(f"Error searching for audio files: {e}")
    
    # Last resort: list available files for manual inspection
    print(f"Available audio files in {voice_overs_folder}:")
    try:
        audio_files = [f for f in voice_overs_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        for audio_file in sorted(audio_files)[:10]:  # Show first 10
            print(f"  • {audio_file.name}")
        if len(audio_files) > 10:
            print(f"  ... and {len(audio_files) - 10} more files")
    except Exception as e:
        print(f"Error listing audio files: {e}")
    
    print(f"Warning: No audio file found for title '{title}'")
    print(f"Suggestion: Check if the audio file exists and has the correct name")
    return None

def load_effects_applied_plan():
    """Load the effects-applied render plan JSON file."""
    effects_plan_path = json_files_dir / config.EFFECTS_APPLIED_JSON
    
    try:
        with open(effects_plan_path, 'r') as f:
            render_plan = json.load(f)
        print(f"✓ Loaded effects-applied render plan: {effects_plan_path}")
        return render_plan
    except FileNotFoundError:
        print(f"Error: Effects-applied render plan not found at {effects_plan_path}")
        print(f"Please run '6. Effects.py' first to generate the processed segments.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in effects-applied render plan: {e}")
        return None

def preprocess_segments_with_padding(segments, output_folder):
    """Pre-process segments to include black screen padding to preserve original timestamps."""
    print(f"Pre-processing segments with timestamp padding...")
    preprocessed_files = []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x["start_time"])
    
    previous_end_time = 0
    
    for i, segment in enumerate(sorted_segments):
        media = segment.get('media', {})
        folder = Path(media.get('folder', ''))
        filename = media.get('file', '')
        start_time = segment.get('start_time', 0)
        segment_duration = media.get('duration', None)
        
        if not filename:
            continue
            
        input_path = folder / filename
        if not input_path.exists():
            print(f"  Warning: File not found: {input_path}")
            continue
        
        # Calculate padding needed before this segment
        padding_duration = start_time - previous_end_time
        
        # Create padded filename
        padded_filename = f"padded_{i+1}_{filename}"
        padded_path = Path(output_folder) / padded_filename
        
        print(f"  Processing segment {i+1}: {filename}")
        print(f"    Original timing: {start_time:.3f}s - {segment.get('end_time', 0):.3f}s")
        print(f"    Padding needed: {padding_duration:.3f}s")
        
        # Build FFmpeg command for this segment with padding
        cmd = ['ffmpeg', '-y']
        
        # Add black padding if needed
        if padding_duration > 0.1:  # Only add padding if > 0.1 seconds
            cmd.extend(['-f', 'lavfi', '-i', f'color=black:size=1920x1080:duration={padding_duration}:rate=24'])
            cmd.extend(['-i', str(input_path)])
            
            # Trim the segment to exact duration if specified
            if segment_duration:
                cmd.extend(['-filter_complex', f'[0:v][1:v]concat=n=2:v=1:a=0[outv]'])
                cmd.extend(['-map', '[outv]'])
                cmd.extend(['-t', str(padding_duration + segment_duration)])
            else:
                cmd.extend(['-filter_complex', f'[0:v][1:v]concat=n=2:v=1:a=0[outv]'])
                cmd.extend(['-map', '[outv]'])
        else:
            # No padding needed, just process the segment
            cmd.extend(['-i', str(input_path)])
            if segment_duration:
                cmd.extend(['-t', str(segment_duration)])
        
        # Output settings
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
        cmd.extend(['-pix_fmt', 'yuv420p'])
        cmd.extend(['-r', '24'])
        cmd.extend(['-s', '1920x1080'])
        cmd.append(str(padded_path))
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                preprocessed_files.append(padded_path)
                print(f"    ✓ Padded segment created: {padded_filename}")
                previous_end_time = segment.get('end_time', start_time + (segment_duration or 0))
            else:
                print(f"    ✗ Failed to create padded segment: {result.stderr}")
                # Fallback to original file without padding
                preprocessed_files.append(input_path)
                previous_end_time = segment.get('end_time', start_time + (segment_duration or 0))
        except Exception as e:
            print(f"    ✗ Error creating padded segment: {e}")
            preprocessed_files.append(input_path)
            previous_end_time = segment.get('end_time', start_time + (segment_duration or 0))
    
    return preprocessed_files

def create_concat_file_from_files(file_paths, output_folder):
    """Create FFmpeg concat file from preprocessed file paths."""
    concat_file_path = Path(output_folder) / "preprocessed_concat_list.txt"
    
    print(f"Creating concat file from preprocessed segments...")
    
    try:
        with open(concat_file_path, 'w') as f:
            for file_path in file_paths:
                f.write(f"file '{str(file_path).replace(chr(92), '/')}'\n")
                print(f"  Added: {Path(file_path).name}")
        
        print(f"✓ Preprocessed concat file created")
        return concat_file_path
        
    except Exception as e:
        print(f"Error creating preprocessed concat file: {e}")
        return None

def check_ffmpeg_cuda():
    """Check if FFmpeg has CUDA support available."""
    try:
        result = subprocess.run(['ffmpeg', '-hwaccels'], 
                               capture_output=True, text=True, timeout=10)
        return 'cuda' in result.stdout.lower()
    except:
        return False

def assemble_final_video(concat_file_path, output_path, audio_file_path=None, dry_run_config=None):
    """Use FFmpeg to concatenate padded segments into final video with preserved timestamps."""
    print(f"\nAssembling final video with preserved timestamps...")
    print(f"Input: {concat_file_path}")
    print(f"Output: {output_path}")
    if audio_file_path:
        print(f"Audio: {audio_file_path}")
        if dry_run_config and dry_run_config['enabled']:
            print(f"🧪 DRY RUN MODE: Will use only first {dry_run_config['duration_seconds']} seconds of audio")
        else:
            print(f"🎵 FULL MODE: Will use complete audio file")
    
    # Build FFmpeg command - back to simple concat but with padded segments
    cmd = ['ffmpeg', '-y']  # -y to overwrite output file
    
    # Check for GPU support
    cuda_available = check_ffmpeg_cuda() and config.USE_GPU_ENCODING
    
    if cuda_available:
        cmd.extend(['-hwaccel', 'cuda'])
        print(f"GPU acceleration: Enabled")
    else:
        print(f"GPU acceleration: Disabled")
    
    # Input concat file (now contains padded segments)
    cmd.extend(['-f', 'concat', '-safe', '0'])
    
    # Add stream loop for full production mode to extend video to match audio
    if audio_file_path and not (dry_run_config and dry_run_config['enabled']):
        cmd.extend(['-stream_loop', '-1'])  # Loop video to match audio duration
        
    cmd.extend(['-i', str(concat_file_path)])
    
    # Add audio input if provided
    if audio_file_path:
        if dry_run_config and dry_run_config['enabled'] and dry_run_config['duration_seconds']:
            cmd.extend(['-ss', '0', '-t', str(dry_run_config['duration_seconds']), '-i', str(audio_file_path)])
            audio_processing = f"Limited to {dry_run_config['duration_seconds']}s (DRY RUN)"
        else:
            cmd.extend(['-i', str(audio_file_path)])
            audio_processing = "Full audio file"
        
        print(f"Audio processing: {audio_processing}")
    
    # Video encoding settings
    if cuda_available:
        cmd.extend(['-c:v', 'h264_nvenc', '-preset', config.OUTPUT_QUALITY])
        encoding_method = f"GPU encoding (h264_nvenc, {config.OUTPUT_QUALITY}) - Padded Concatenation"
    else:
        cmd.extend(['-c:v', 'libx264', '-preset', config.OUTPUT_QUALITY])
        encoding_method = f"CPU encoding (libx264, {config.OUTPUT_QUALITY}) - Padded Concatenation"
    
    # Additional settings
    cmd.extend(['-pix_fmt', 'yuv420p'])
    cmd.extend(['-r', '24'])
    
    # Audio settings if present
    if audio_file_path:
        cmd.extend(['-c:a', 'aac'])
        if dry_run_config and dry_run_config['enabled'] and dry_run_config['duration_seconds']:
            cmd.extend(['-t', str(dry_run_config['duration_seconds'])])
        else:
            # For full production mode, video will loop to match audio duration
            cmd.extend(['-shortest'])  # Stop when shortest stream (audio) ends
    
    cmd.append(str(output_path))
    
    print(f"Encoding method: {encoding_method}")
    print(f"FFmpeg command: {' '.join(cmd[:6])}... [padded concat] -> {Path(output_path).name}")
    
    try:
        # Run FFmpeg
        result = subprocess.run(
            cmd, 
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ Final video assembled with preserved timestamps!")
            return True
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"✗ FFmpeg failed with return code {result.returncode}")
            print(f"Error: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ FFmpeg assembly timed out after 10 minutes")
        return False
    except FileNotFoundError:
        print(f"✗ FFmpeg not found. Please install FFmpeg.")
        return False
    except Exception as e:
        print(f"✗ FFmpeg assembly failed: {e}")
        return False

def create_visual_timeline(effects_plan):
    """Create a visual timeline showing EXACTLY how the final video will be assembled."""
    try:
        timeline_file = "JSON Files/Timeline Preview.txt"
        
        # Get transcription data
        transcriptions = {}
        try:
            with open('JSON Files/2. Transcriptions.json', 'r', encoding='utf-8') as f:
                trans_data = json.load(f)
                file_path = next(iter(trans_data['transcriptions']))
                transcriptions = trans_data['transcriptions'][file_path]['segments']
        except Exception as e:
            print(f"Warning: Could not load transcriptions: {e}")
        
        # Get segments from the effects applied plan
        segments = effects_plan.get('segments', [])
        if not segments:
            log("No segments found in render plan", "error")
            return None
            
        # Sort segments by their ORIGINAL start time (as they appear in JSON)
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])
        
        # Create the ACTUAL ASSEMBLY timeline (sequential concatenation)
        # This is how the video will ACTUALLY be assembled
        assembly_segments = []
        current_assembly_time = 0.0
        
        for segment in sorted_segments:
            # Each segment will be placed sequentially in the final video
            segment_duration = segment.get('end_time', 0) - segment.get('start_time', 0)
            
            assembly_segment = segment.copy()
            # ACTUAL assembly timing (sequential concatenation)
            assembly_segment['assembly_start'] = current_assembly_time
            assembly_segment['assembly_end'] = current_assembly_time + segment_duration
            assembly_segment['assembly_duration'] = segment_duration
            
            # Keep original JSON timing for reference
            assembly_segment['original_start'] = segment['start_time']
            assembly_segment['original_end'] = segment['end_time']
            
            assembly_segments.append(assembly_segment)
            current_assembly_time += segment_duration
        
        # Create the timeline
        timeline = []
        timeline.append("\n" + "=" * 120)
        timeline.append("🎬 FINAL VIDEO ASSEMBLY TIMELINE (ACTUAL OUTPUT)")
        timeline.append("=" * 120 + "\n")
        
        # Add video info
        timeline.append(f"Video: {effects_plan['video_info']['title']}")
        timeline.append(f"Preview Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        timeline.append(f"Output File: {AssemblyConfig.FINAL_VIDEO_NAME}")
        timeline.append(f"Encoding: {'GPU (NVENC)' if AssemblyConfig.USE_GPU_ENCODING else 'CPU'}")
        timeline.append(f"Quality Preset: {AssemblyConfig.OUTPUT_QUALITY}")
        timeline.append(f"Assembly Method: Sequential Concatenation")
        timeline.append("\n" + "=" * 120 + "\n")
        timeline.append("ACTUAL ASSEMBLY TIMELINE (Sequential Order)")
        timeline.append("-" * 120 + "\n")
        
        # Display segments using ACTUAL assembly timing
        for i, segment in enumerate(assembly_segments):
            assembly_start = segment['assembly_start']
            assembly_end = segment['assembly_end']
            original_start = segment['original_start']
            original_end = segment['original_end']
            celebrity = segment['celebrity']
            media_type = segment['media']['type']
            filename = segment['media']['file']
            duration = segment['assembly_duration']
            
            # Format timestamps (MM:SS.ms) using ACTUAL assembly times
            start_min = int(assembly_start // 60)
            start_sec = assembly_start % 60
            end_min = int(assembly_end // 60)
            end_sec = assembly_end % 60
            
            start_str = f"{start_min:02d}:{start_sec:06.3f}"
            end_str = f"{end_min:02d}:{end_sec:06.3f}"
            
            # Format original timestamps for reference
            orig_start_min = int(original_start // 60)
            orig_start_sec = original_start % 60
            orig_end_min = int(original_end // 60)
            orig_end_sec = original_end % 60
            
            orig_start_str = f"{orig_start_min:02d}:{orig_start_sec:06.3f}"
            orig_end_str = f"{orig_end_min:02d}:{orig_end_sec:06.3f}"
            
            # Find corresponding transcription text using ORIGINAL timestamps
            segment_text = ""
            for trans in transcriptions:
                if abs(trans['start'] - original_start) < 2.0:  # Within 2 seconds of original timing
                    segment_text = trans['text'].strip()
                    break
            
            # Create the visual representation
            timeline.append(f"SEGMENT {i+1}: [{start_str} → {end_str}] ({duration:.3f}s)")
            
            # Celebrity name bar with media type and effects indicators
            icon = "🖼️" if media_type == "image" else "🎥"
            effects = []
            if "effects_applied" in segment and segment["effects_applied"]:
                for effect in segment["effects_applied"]:
                    if effect == "split_screen":
                        effects.append("Split Screen")
                    elif effect == "ken_burns":
                        effects.append("Ken Burns")
                    elif effect == "composite":
                        effects.append("Composite")
            
            effects_str = f" ({', '.join(effects)})" if effects else ""
            
            # Handle split screen celebrities display
            if "split_screen_celebrities" in segment:
                split_celebs = segment["split_screen_celebrities"]
                celebrity_display = f"{split_celebs[0]} & {split_celebs[1]}"
            else:
                celebrity_display = celebrity
                
            celebrity_bar = f"{icon} {celebrity_display.upper()}{effects_str}"
            timeline.append(f"    {celebrity_bar:-^100}")
            
            # Media file info with original timing reference
            timeline.append(f"    Duration: {duration:.3f}s | File: {filename}")
            timeline.append(f"    Original Audio Timing: [{orig_start_str} → {orig_end_str}]")
            
            # Transcription text (if available)
            if segment_text:
                import textwrap
                wrapped_text = textwrap.wrap(segment_text, width=90)
                for line in wrapped_text:
                    timeline.append(f"    💬 {line}")
            
            timeline.append(f"    {'-' * 100}\n")
        
        # Add statistics
        timeline.append("ASSEMBLY STATISTICS")
        timeline.append("-" * 120 + "\n")
        
        # Group by celebrity (handling split screen)
        celebrity_stats = {}
        total_assembly_duration = max(segment['assembly_end'] for segment in assembly_segments) if assembly_segments else 0
        
        for segment in assembly_segments:
            celebrity = segment['celebrity']
            duration = segment['assembly_duration']
            media_type = segment['media']['type']
            
            # Handle split screen celebrities
            if "split_screen_celebrities" in segment:
                # Count for both celebrities in split screen
                for celeb in segment["split_screen_celebrities"]:
                    if celeb not in celebrity_stats:
                        celebrity_stats[celeb] = {
                            "total_duration": 0,
                            "video_count": 0,
                            "image_count": 0,
                            "effects_count": 0
                        }
                    
                    celebrity_stats[celeb]["total_duration"] += duration
                    if media_type == "video":
                        celebrity_stats[celeb]["video_count"] += 1
                    else:
                        celebrity_stats[celeb]["image_count"] += 1
                    
                    if "effects_applied" in segment and segment["effects_applied"]:
                        celebrity_stats[celeb]["effects_count"] += len(segment["effects_applied"])
            else:
                # Regular single celebrity segment
                if celebrity not in celebrity_stats:
                    celebrity_stats[celebrity] = {
                        "total_duration": 0,
                        "video_count": 0,
                        "image_count": 0,
                        "effects_count": 0
                    }
                
                celebrity_stats[celebrity]["total_duration"] += duration
                if media_type == "video":
                    celebrity_stats[celebrity]["video_count"] += 1
                else:
                    celebrity_stats[celebrity]["image_count"] += 1
                
                if "effects_applied" in segment and segment["effects_applied"]:
                    celebrity_stats[celebrity]["effects_count"] += len(segment["effects_applied"])
        
        # Display statistics
        for celebrity, stats in celebrity_stats.items():
            percentage = (stats["total_duration"] / total_assembly_duration * 100) if total_assembly_duration > 0 else 0
            timeline.append(f"{celebrity}:")
            timeline.append(f"  • Screen Time: {stats['total_duration']:.3f}s ({percentage:.1f}%)")
            timeline.append(f"  • Media: {stats['video_count']} videos, {stats['image_count']} images")
            timeline.append(f"  • Effects Applied: {stats['effects_count']}")
            timeline.append("")
        
        timeline.append(f"Total Final Video Duration: {total_assembly_duration:.3f} seconds")
        timeline.append(f"Total Segments: {len(assembly_segments)}")
        timeline.append("")
        timeline.append("IMPORTANT: This timeline shows the ACTUAL final video output.")
        timeline.append("Segments are concatenated sequentially, not placed at original audio timestamps.")
        timeline.append("The 'Original Audio Timing' shows where this content was mentioned in the source audio.")
        timeline.append("")
        timeline.append("=" * 120)
        
        # Save the timeline
        with open(timeline_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(timeline))
        
        # Also print to console
        print("\n".join(timeline))
        print(f"\nActual assembly timeline saved to: {timeline_file}")
        
        return timeline_file
        
    except Exception as e:
        print(f"Error creating visual timeline: {e}")
        return None

def main():
    """Main function to assemble the final video."""
    try:
        log("🎬 FINAL VIDEO ASSEMBLY", "header")
        print("=" * 50)
        
        # Load the effects-applied render plan
        effects_plan = load_effects_applied_plan()
        if not effects_plan:
            return False
        
        # Create and display visual timeline
        log("Creating visual timeline preview...", "wait")
        timeline_file = create_visual_timeline(effects_plan)
        
        # Wait for 5 seconds to let user review the timeline
        print("\nReviewing timeline for 5 seconds...")
        time.sleep(5)
        
        # Continue with existing assembly process
        log("Starting video assembly...", "header")
        print("=" * 50)
        
        # Load dry run configuration first
        dry_run_config = load_transcriptions_config()
        print()
        
        # Load detections to get video title
        detections = load_detections()
        if detections:
            video_info = detections.get('video', {})
            title = video_info.get('title', 'Unknown')
            print(f"Video title from detections: '{title}'")
        else:
            # Fallback to render plan title
            video_info = effects_plan.get('video_info', {})
            title = video_info.get('title', 'Unknown')
            print(f"Using title from render plan: '{title}'")
        
        # Get segments
        segments = effects_plan.get('segments', [])
        if not segments:
            print("No segments found in render plan. Nothing to assemble.")
            return
        
        print(f"Found {len(segments)} segments to assemble:")
        
        # Create output folder
        output_folder = Path(config.OUTPUT_FOLDER)
        output_folder.mkdir(exist_ok=True)
        
        # Pre-process segments with padding to preserve timestamps
        preprocessed_files = preprocess_segments_with_padding(segments, output_folder)
        
        # Create concat file from preprocessed segments
        concat_file_path = create_concat_file_from_files(preprocessed_files, output_folder)
        if not concat_file_path:
            print("Failed to create concat file. Exiting.")
            return
        
        # Define final output path
        final_output_path = output_folder / config.FINAL_VIDEO_NAME
        
        # Find audio file
        audio_file_path = find_audio_file(title)
        
        # Display assembly configuration summary
        print(f"\n{'='*60}")
        print(f"ASSEMBLY CONFIGURATION")
        print(f"{'='*60}")
        print(f"📹 Video segments: {len(segments)} segments")
        print(f"🎵 Audio file: {audio_file_path.name if audio_file_path else 'None found'}")
        if dry_run_config['enabled']:
            print(f"🧪 Processing mode: DRY RUN")
            print(f"⏱️  Audio duration: {dry_run_config['duration_seconds']} seconds (limited)")
            print(f"💡 Purpose: Testing/preview")
        else:
            print(f"🎬 Processing mode: FULL PRODUCTION")
            print(f"⏱️  Audio duration: Complete file")
            print(f"💡 Purpose: Final production")
        print(f"⚡ GPU acceleration: {'Enabled' if check_ffmpeg_cuda() and config.USE_GPU_ENCODING else 'Disabled'}")
        print(f"📁 Output folder: {config.OUTPUT_FOLDER}")
        print(f"{'='*60}")
        
        # Assemble final video
        success = assemble_final_video(concat_file_path, final_output_path, audio_file_path, dry_run_config)
        
        if success:
            # Get video info
            video_info = effects_plan.get('video_info', {})
            title = video_info.get('title', 'Unknown')
            
            print(f"\n{'='*60}")
            print(f"FINAL ASSEMBLY COMPLETE! 🎬")
            print(f"{'='*60}")
            print(f"📹 Title: {title}")
            print(f"📁 Output: {final_output_path}")
            print(f"🎞️  Segments: {len(segments)} processed segments")
            print(f"⚡ Method: FFmpeg concat ({'GPU accelerated' if check_ffmpeg_cuda() and config.USE_GPU_ENCODING else 'CPU'})")
            
            # Display dry run mode information
            if dry_run_config['enabled']:
                print(f"🧪 Mode: DRY RUN - Audio limited to {dry_run_config['duration_seconds']} seconds")
                print(f"💡 Purpose: Testing/preview mode")
            else:
                print(f"🎵 Mode: FULL PRODUCTION - Complete audio used")
                print(f"💡 Purpose: Final production mode")
            
            print(f"💾 Size: {final_output_path.stat().st_size / (1024*1024):.1f} MB" if final_output_path.exists() else "")
            
            # Clean up concat file
            try:
                concat_file_path.unlink()
                print(f"🧹 Cleaned up temporary concat file")
            except:
                pass
            
            print(f"\n🎉 Your final video is ready! ✨")
            
        else:
            print(f"\n❌ Final assembly failed!")
            print(f"Check the error messages above for troubleshooting.")
        
        return True
        
    except Exception as e:
        print(f"Error in main function: {e}")
        return False

if __name__ == "__main__":
    main() 
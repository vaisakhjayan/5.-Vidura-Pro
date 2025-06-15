#!/usr/bin/env python3
"""
Audio Transcription Module with Word-Level Timestamps and Notion Script Integration
===================================================================================
This module transcribes audio files using faster-whisper and whisperX
to provide accurate transcriptions with word-by-word timestamps.
It also integrates with Notion to fetch the actual script content for perfect accuracy.

🔧 DRY RUN MODE CONFIGURATION:
- DRY_RUN_MODE: Set to True to enable dry run (transcribe only a portion of audio)
- DRY_RUN_DURATION: Duration in seconds to transcribe when dry run is enabled (default: 30)

Requirements:
- faster-whisper
- whisperx
- torch
- torchaudio
- notion-client
"""

# ================================
# DRY RUN MODE CONFIGURATION
# ================================
# DRY_RUN_MODE will be fetched from Notion Database: 20302cd2c1428027bb04f1d147b50cf9
# Property: "Dry Run Mode" (Select: False/True)
DRY_RUN_DURATION = 60  # Duration in seconds to transcribe when in dry run mode
NOTION_CONFIG_DATABASE_ID = "20302cd2c1428027bb04f1d147b50cf9"  # Notion database for configuration

# Global variable that will be set from Notion database
DRY_RUN_MODE = False  # Will be overridden by Notion fetch in main()

# Global variable for composite clips interval (used by Effects.py)
COMPOSITE_APPLY_INTERVAL = 3  # Will be overridden by Notion fetch in main()

# Global variable for split screen effect control (used by Effects.py)
SPLIT_SCREEN_ENABLED = True  # Will be overridden by Notion fetch in main()
# ================================

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import datetime
from difflib import SequenceMatcher

try:
    from faster_whisper import WhisperModel
    import whisperx
    import torch
    from notion_client import Client
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    print("Run: pip install faster-whisper whisperx torch torchaudio notion-client")
    exit(1)

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
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
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

def save_text_file(file_path: str, content: str):
    """Save text content to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_readable_transcription_report(export_data: dict) -> str:
    """Create a readable text format of the transcription data"""
    
    # Extract metadata
    export_metadata = export_data.get("export_metadata", {})
    transcriptions = export_data.get("transcriptions", {})
    
    # Start building the report
    report = []
    report.append("=" * 80)
    report.append("AUDIO TRANSCRIPTION REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Export metadata section
    report.append(f"Export Timestamp: {export_metadata.get('export_timestamp', 'N/A')}")
    report.append(f"Total Files Processed: {export_metadata.get('total_files_processed', 0)}")
    report.append(f"Exported From: {export_metadata.get('exported_from', 'N/A')}")
    report.append("")
    
    # Dry run mode info
    dry_run_info = export_metadata.get("dry_run_mode", {})
    if dry_run_info.get("enabled", False):
        report.append("🧪 PROCESSING MODE: DRY RUN")
        report.append(f"   • Duration: {dry_run_info.get('duration_seconds', 'N/A')} seconds")
        report.append(f"   • Status: {dry_run_info.get('status', 'N/A')}")
        report.append(f"   • Source: {dry_run_info.get('source', 'N/A')}")
    else:
        report.append("🎬 PROCESSING MODE: FULL TRANSCRIPTION")
        report.append("   • Complete audio file processed")
    report.append("")
    
    # Models used
    models_info = export_metadata.get("models_used", {})
    report.append("🤖 MODELS USED:")
    report.append(f"   • Faster Whisper: {models_info.get('faster_whisper', 'N/A')}")
    report.append(f"   • WhisperX Alignment: {models_info.get('whisperx_alignment', 'N/A')}")
    report.append("")
    
    report.append("-" * 80)
    report.append("TRANSCRIPTION DETAILS")
    report.append("-" * 80)
    report.append("")
    
    # Process each transcription file
    for file_path, transcription_data in transcriptions.items():
        # File information
        metadata = transcription_data.get("metadata", {})
        file_name = metadata.get("file_name", "Unknown File")
        
        report.append(f"📁 FILE: {file_name}")
        report.append(f"📍 PATH: {file_path}")
        report.append("")
        
        # Processing information
        processing_time = metadata.get("processing_time_seconds", 0)
        channel_name = metadata.get("channel_name", "N/A")
        keywords_used = metadata.get("keywords_used", 0)
        
        report.append("📊 PROCESSING INFO:")
        report.append(f"   • Processing Time: {processing_time:.2f} seconds")
        report.append(f"   • Channel: {channel_name}")
        report.append(f"   • Keywords Used: {keywords_used}")
        
        # Notion integration info
        notion_info = metadata.get("notion_integration", {})
        if notion_info.get("script_aligned", False):
            report.append(f"   • Notion Script: ✅ Aligned (Page ID: {notion_info.get('page_id', 'N/A')})")
        else:
            report.append("   • Notion Script: ❌ Not aligned")
        
        report.append("")
        
        # Language detection
        language = transcription_data.get("language", "N/A")
        language_prob = transcription_data.get("language_probability", 0)
        report.append(f"🌐 LANGUAGE: {language.upper()} (Confidence: {language_prob:.1%})")
        report.append("")
        
        # Segments timeline
        segments = transcription_data.get("segments", [])
        if segments:
            report.append("-" * 60)
            report.append("TRANSCRIPTION TIMELINE")
            report.append("-" * 60)
            report.append("")
            
            total_duration = 0
            word_count = 0
            
            for i, segment in enumerate(segments, 1):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                # Calculate duration and word count
                duration = end_time - start_time
                total_duration += duration
                word_count += len(text.split()) if text else 0
                
                # Format timestamps
                start_min = int(start_time // 60)
                start_sec = start_time % 60
                end_min = int(end_time // 60)
                end_sec = end_time % 60
                
                if start_min > 0:
                    start_formatted = f"{start_min}:{start_sec:05.2f}"
                else:
                    start_formatted = f"{start_sec:.2f}"
                
                if end_min > 0:
                    end_formatted = f"{end_min}:{end_sec:05.2f}"
                else:
                    end_formatted = f"{end_sec:.2f}"
                
                # Show segment
                report.append(f"{i:3d}. [{start_formatted} → {end_formatted}] ({duration:.2f}s)")
                report.append(f"     {text}")
                
                # Show correction info if available
                if segment.get("was_corrected", False):
                    original = segment.get("original_transcription", "")
                    if original and original != text:
                        report.append(f"     📝 Original: {original}")
                
                report.append("")
            
            # Statistics
            report.append("-" * 60)
            report.append("TRANSCRIPTION STATISTICS")
            report.append("-" * 60)
            report.append("")
            
            report.append(f"📈 SUMMARY:")
            report.append(f"   • Total Segments: {len(segments)}")
            report.append(f"   • Total Duration: {total_duration:.2f} seconds")
            report.append(f"   • Total Words: {word_count}")
            report.append(f"   • Average Segment Length: {total_duration/len(segments):.2f} seconds")
            report.append(f"   • Words Per Minute: {(word_count / total_duration * 60):.1f} WPM")
            
            # Check for corrections
            corrected_segments = sum(1 for seg in segments if seg.get("was_corrected", False))
            if corrected_segments > 0:
                correction_rate = (corrected_segments / len(segments)) * 100
                report.append(f"   • Segments Corrected: {corrected_segments}/{len(segments)} ({correction_rate:.1f}%)")
            
            # Word-level statistics
            total_words_with_timestamps = 0
            for segment in segments:
                words = segment.get("words", [])
                if isinstance(words, list):
                    total_words_with_timestamps += len(words)
            
            if total_words_with_timestamps > 0:
                report.append(f"   • Words with Timestamps: {total_words_with_timestamps}")
            
            report.append("")
            
            # Clean transcription section
            report.append("=" * 80)
            report.append("CLEAN TRANSCRIPTION (NO TIMESTAMPS)")
            report.append("=" * 80)
            report.append("")
            
            # Extract just the text from all segments
            clean_text_parts = []
            for segment in segments:
                text = segment.get("text", "").strip()
                if text:
                    clean_text_parts.append(text)
            
            # Join all text and format nicely
            clean_transcription = " ".join(clean_text_parts)
            
            # Add line breaks for readability (every ~80 characters at word boundaries)
            formatted_lines = []
            words = clean_transcription.split()
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > 80 and current_line:
                    formatted_lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += word_length
            
            if current_line:
                formatted_lines.append(" ".join(current_line))
            
            for line in formatted_lines:
                report.append(line)
            
            report.append("")
        
        report.append("=" * 80)
        report.append("End of Transcription Report")
        report.append("=" * 80)
    
    return "\n".join(report)

def fetch_configuration_from_notion(notion_token: str) -> tuple[bool, int, bool]:
    """
    Fetch configuration settings from the Notion configuration database.
    
    Args:
        notion_token: Notion integration token
        
    Returns:
        Tuple of (dry_run_mode: bool, composite_clips_interval: int, split_screen_enabled: bool)
        Defaults to (False, 3, True) if unable to fetch
    """
    try:
        if not notion_token:
            log("No Notion token provided, using default settings: Dry Run=False, Composite=3, Split Screen=True", "warn")
            return False, 3, True
        
        # Initialize Notion client
        notion_client = Client(auth=notion_token)
        
        # Query the database to get the configuration settings
        response = notion_client.databases.query(database_id=NOTION_CONFIG_DATABASE_ID)
        
        if not response.get("results"):
            log("No records found in Notion configuration database", "warn")
            return False, 3, True
        
        # Get the first record (assuming single configuration record)
        record = response["results"][0]
        properties = record.get("properties", {})
        
        # Extract the "Dry Run Mode" property
        dry_run_mode = False
        dry_run_property = properties.get("Dry Run Mode", {})
        if dry_run_property.get("type") == "select":
            dry_run_value = dry_run_property.get("select", {}).get("name", "False")
            dry_run_mode = dry_run_value.lower() == "true"
        
        # Extract the "Composite Clips" property
        composite_interval = 3  # Default value
        composite_property = properties.get("Composite Clips", {})
        if composite_property.get("type") == "select":
            composite_value = composite_property.get("select", {}).get("name", "3")
            try:
                composite_interval = int(composite_value)
            except (ValueError, TypeError):
                log(f"Invalid Composite Clips value '{composite_value}', using default 3", "warn")
                composite_interval = 3
        
        # Extract the "Split Screen" property
        split_screen_enabled = True  # Default value
        split_screen_property = properties.get("Split Screen", {})
        if split_screen_property.get("type") == "select":
            split_screen_value = split_screen_property.get("select", {}).get("name", "ON")
            split_screen_enabled = split_screen_value.upper() == "ON"
        
        log(f"📡 Fetched from Notion:", "info")
        log(f"   • Dry Run Mode = {dry_run_value if 'dry_run_value' in locals() else 'False'} ({dry_run_mode})", "info")
        log(f"   • Composite Clips = {composite_value if 'composite_value' in locals() else '3'} ({composite_interval})", "info")
        log(f"   • Split Screen = {split_screen_value if 'split_screen_value' in locals() else 'ON'} ({split_screen_enabled})", "info")
        
        return dry_run_mode, composite_interval, split_screen_enabled
            
    except Exception as e:
        log(f"Error fetching configuration from Notion: {e}", "error")
        log("Falling back to default settings: Dry Run=False, Composite=3, Split Screen=True", "warn")
        return False, 3, True

def get_composite_apply_interval() -> int:
    """
    Get the current composite apply interval value.
    This function can be imported by other scripts (like Effects.py) to get the Notion-configured value.
    
    Returns:
        Integer value for composite apply interval (defaults to 3)
    """
    return COMPOSITE_APPLY_INTERVAL

def initialize_notion_config(notion_token: str = None) -> tuple[bool, int, bool]:
    """
    Initialize the global configuration variables from Notion.
    This can be called by other scripts to load the Notion configuration.
    
    Args:
        notion_token: Notion integration token (optional, uses default if not provided)
        
    Returns:
        Tuple of (dry_run_mode: bool, composite_clips_interval: int, split_screen_enabled: bool)
    """
    global DRY_RUN_MODE, COMPOSITE_APPLY_INTERVAL, SPLIT_SCREEN_ENABLED
    
    if notion_token is None:
        notion_token = "ntn_cC7520095381SElmcgTOADYsGnrABFn2ph1PrcaGSst2dv"  # Default token
    
    DRY_RUN_MODE, COMPOSITE_APPLY_INTERVAL, SPLIT_SCREEN_ENABLED = fetch_configuration_from_notion(notion_token)
    return DRY_RUN_MODE, COMPOSITE_APPLY_INTERVAL, SPLIT_SCREEN_ENABLED

class AudioTranscriber:
    """
    Audio transcription class using faster-whisper and whisperX
    for high-quality transcription with word-level timestamps.
    """
    
    def __init__(self, 
                 faster_whisper_model: str = "large-v3",
                 device: str = "auto",
                 compute_type: str = "float16",
                 notion_token: str = None):
        """
        Initialize the transcriber with model configurations and Notion integration.
        
        Args:
            faster_whisper_model: Model size for faster-whisper
            device: Device to use (auto, cpu, cuda)
            compute_type: Computation type (float16, int8, float32)
            notion_token: Notion integration token (optional)
        """
        self.faster_whisper_model = faster_whisper_model
        self.device = self._get_device(device)
        self.compute_type = compute_type
        
        # Initialize models
        self.whisper_model = None
        self.alignment_model = None
        self.metadata = None
        
        # Initialize Notion client
        self.notion_client = None
        if notion_token:
            try:
                self.notion_client = Client(auth=notion_token)
            except Exception as e:
                log(f"Failed to initialize Notion client: {e}", "warn")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_models(self):
        """Load faster-whisper and whisperX models."""
        try:
            # Try to load the model with specified configuration
            try:
                self.whisper_model = WhisperModel(
                    self.faster_whisper_model,
                    device=self.device,
                    compute_type=self.compute_type
                )
            except Exception as cuda_error:
                if "cuda" in str(cuda_error).lower() or "cudnn" in str(cuda_error).lower():
                    log(f"CUDA/cuDNN error encountered: {cuda_error}", "warn")
                    log("Falling back to CPU mode...", "info")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.whisper_model = WhisperModel(
                        self.faster_whisper_model,
                        device=self.device,
                        compute_type=self.compute_type
                    )
                else:
                    raise cuda_error
            
            # Load alignment model
            try:
                self.alignment_model, self.metadata = whisperx.load_align_model(
                    language_code="en", 
                    device=self.device
                )
            except Exception as align_error:
                if "cuda" in str(align_error).lower() or "cudnn" in str(align_error).lower():
                    log(f"CUDA/cuDNN error in alignment model: {align_error}", "warn")
                    log("Loading alignment model with CPU...", "info")
                    self.alignment_model, self.metadata = whisperx.load_align_model(
                        language_code="en", 
                        device="cpu"
                    )
                else:
                    raise align_error
            
        except Exception as e:
            log(f"Error loading models: {e}", "error")
            raise
    
    def get_supported_audio_formats(self) -> List[str]:
        """Return list of supported audio formats."""
        return ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma']
    
    def find_audio_files(self, folder_path: str) -> List[Path]:
        """Find all audio files in the specified folder."""
        folder = Path(folder_path)
        if not folder.exists():
            log(f"Folder does not exist: {folder_path}", "error")
            return []
        
        audio_files = []
        supported_formats = self.get_supported_audio_formats()
        
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                audio_files.append(file_path)
        
        return audio_files
    
    def transcribe_with_faster_whisper(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using faster-whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            # Check if dry run mode is enabled
            if DRY_RUN_MODE:
                log(f"🧪 DRY RUN MODE: Transcribing only first {DRY_RUN_DURATION} seconds of: {audio_path}", "warn")
                # Use clip_timestamps to limit transcription duration
                segments, info = self.whisper_model.transcribe(
                    audio_path,
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False,
                    clip_timestamps=[0, DRY_RUN_DURATION]  # Transcribe only first N seconds
                )
            else:
                log(f"Transcribing with faster-whisper: {audio_path}", "wait")
                segments, info = self.whisper_model.transcribe(
                    audio_path,
                    beam_size=5,
                    language="en",
                    condition_on_previous_text=False
                )
            
            # Convert segments to list for whisperX compatibility
            segments_list = []
            for segment in segments:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
            
            return {
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability
            }
            
        except Exception as e:
            error_str = str(e).lower()
            if "cudnn" in error_str or "cuda" in error_str:
                log(f"CUDA/cuDNN error during transcription: {e}", "warn")
                log("Reloading model with CPU and retrying transcription...", "info")
                
                # Reload model with CPU
                self.device = "cpu"
                self.compute_type = "int8"
                self.whisper_model = WhisperModel(
                    self.faster_whisper_model,
                    device=self.device,
                    compute_type=self.compute_type
                )
                
                # Retry transcription with CPU (with dry run support)
                if DRY_RUN_MODE:
                    log(f"🧪 DRY RUN MODE: Retrying with CPU - transcribing only first {DRY_RUN_DURATION} seconds", "warn")
                    segments, info = self.whisper_model.transcribe(
                        audio_path,
                        beam_size=5,
                        language="en",
                        condition_on_previous_text=False,
                        clip_timestamps=[0, DRY_RUN_DURATION]
                    )
                else:
                    segments, info = self.whisper_model.transcribe(
                        audio_path,
                        beam_size=5,
                        language="en",
                        condition_on_previous_text=False
                    )
                
                # Convert segments to list for whisperX compatibility
                segments_list = []
                for segment in segments:
                    segments_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
                
                log("Successfully completed transcription with CPU fallback", "success")
                
                return {
                    "segments": segments_list,
                    "language": info.language,
                    "language_probability": info.language_probability
                }
            else:
                log(f"Error in faster-whisper transcription: {e}", "error")
                raise
    
    def load_channel_keywords(self, channel_name: str) -> List[str]:
        """Load keywords from the channel's JSON file that need word-level timestamps."""
        try:
            channel_file_path = f"Channel JSON Files/{channel_name}.json"
            
            if not os.path.exists(channel_file_path):
                return []
            
            with open(channel_file_path, 'r', encoding='utf-8') as f:
                channel_data = json.load(f)
            
            # Extract all keywords from both channel-specific and general words
            all_keywords = []
            
            # Get channel-specific keywords
            if channel_name in channel_data:
                for person_name, keywords in channel_data[channel_name].items():
                    all_keywords.extend(keywords)
            
            # Get general words
            if "Words" in channel_data:
                for category, keywords in channel_data["Words"].items():
                    all_keywords.extend(keywords)
            
            # Remove duplicates and convert to lowercase for matching
            unique_keywords = list(set([keyword.lower() for keyword in all_keywords]))
            return unique_keywords
            
        except Exception as e:
            return []
    
    def extract_keyword_segments(self, segments: List[Dict], keywords: List[str]) -> List[Dict]:
        """
        Identify segments that contain keywords and need word-level timestamps.
        
        Args:
            segments: List of segment dictionaries from faster-whisper
            keywords: List of keywords to search for
            
        Returns:
            List of segments that contain keywords
        """
        keyword_segments = []
        
        for segment in segments:
            segment_text_lower = segment["text"].lower()
            
            # Check if any keyword appears in this segment
            for keyword in keywords:
                if keyword in segment_text_lower:
                    keyword_segments.append(segment)
                    break  # Found at least one keyword, no need to check others
        
        log(f"Found {len(keyword_segments)} segments containing keywords out of {len(segments)} total segments", "info")
        return keyword_segments
    
    def add_selective_word_timestamps(self, audio_path: str, transcription_result: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Add word-level timestamps only for segments containing keywords."""
        try:
            segments = transcription_result["segments"]
            
            # Find segments that contain keywords
            keyword_segments = self.extract_keyword_segments(segments, keywords)
            
            if not keyword_segments:
                return transcription_result
            
            # Load audio for whisperX (only if we have keywords to process)
            audio = whisperx.load_audio(audio_path)
            
            try:
                # Only align the segments that contain keywords
                result_aligned = whisperx.align(
                    keyword_segments, 
                    self.alignment_model, 
                    self.metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
                
                if not result_aligned or "segments" not in result_aligned:
                    return transcription_result
                
                # Create a mapping of original segments to aligned segments
                aligned_segments_dict = {}
                for aligned_seg in result_aligned["segments"]:
                    # Find matching original segment
                    for orig_seg in keyword_segments:
                        if abs(orig_seg["start"] - aligned_seg["start"]) < 0.1:  # 100ms tolerance
                            aligned_segments_dict[orig_seg["start"]] = aligned_seg
                            break
                
                # Update original segments with word-level data where available
                for i, segment in enumerate(segments):
                    if segment["start"] in aligned_segments_dict:
                        aligned_seg = aligned_segments_dict[segment["start"]]
                        # Preserve the original text but add word timestamps
                        segments[i] = {
                            **segment,
                            "words": aligned_seg.get("words", [])
                        }
            
            except Exception as e:
                log(f"Error in whisperX alignment: {e}", "error")
                # Continue with original segments if alignment fails
            
            transcription_result["segments"] = segments
            return transcription_result
            
        except Exception as e:
            log(f"Error adding selective word timestamps: {e}", "error")
            return transcription_result
    
    def transcribe_audio_file(self, audio_path: str, channel_name: str = None, page_id: str = None) -> Dict[str, Any]:
        """Complete transcription pipeline for a single audio file with Notion script integration."""
        start_time = time.time()
        
        try:
            # Step 1: Transcribe with faster-whisper
            transcription = self.transcribe_with_faster_whisper(audio_path)
            
            # Step 2: Load keywords and add selective word-level timestamps
            keywords = []
            if channel_name:
                keywords = self.load_channel_keywords(channel_name)
            
            if keywords:
                enhanced_transcription = self.add_selective_word_timestamps(audio_path, transcription, keywords)
            else:
                enhanced_transcription = transcription
            
            # Step 3: Fetch and align with Notion script for perfect accuracy
            script_aligned = False
            if page_id and self.notion_client:
                script_text = self.fetch_notion_script(page_id)
                if script_text:
                    enhanced_transcription["segments"] = self.align_script_with_transcription(
                        script_text, enhanced_transcription["segments"]
                    )
                    enhanced_transcription["notion_script"] = {
                        "page_id": page_id,
                        "script_length": len(script_text),
                        "alignment_applied": True
                    }
                    script_aligned = True
            
            # Add metadata
            processing_time = time.time() - start_time
            enhanced_transcription["metadata"] = {
                "file_path": str(audio_path),
                "file_name": Path(audio_path).name,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.datetime.now().isoformat(),
                "channel_name": channel_name,
                "keywords_used": len(keywords) if keywords else 0,
                "dry_run_mode": {
                    "enabled": DRY_RUN_MODE,
                    "duration_seconds": DRY_RUN_DURATION if DRY_RUN_MODE else None,
                    "processing_status": f"DRY RUN MODE - Transcribed {DRY_RUN_DURATION} seconds only" if DRY_RUN_MODE else "FULL MODE - Complete audio transcribed",
                    "source": "notion_database",
                    "database_id": NOTION_CONFIG_DATABASE_ID,
                    "note": f"Only first {DRY_RUN_DURATION} seconds transcribed (from Notion)" if DRY_RUN_MODE else "Full audio transcribed"
                },
                "composite_clips_config": {
                    "apply_interval": COMPOSITE_APPLY_INTERVAL,
                    "source": "notion_database",
                    "database_id": NOTION_CONFIG_DATABASE_ID,
                    "note": f"Apply composite effect every {COMPOSITE_APPLY_INTERVAL} clips (from Notion)"
                },
                "notion_integration": {
                    "page_id": page_id,
                    "script_aligned": script_aligned
                },
                "models_used": {
                    "faster_whisper": self.faster_whisper_model,
                    "whisperx_alignment": "en" if keywords else "not_used",
                    "notion_script_integration": "enabled" if script_aligned else "disabled"
                }
            }
            
            return enhanced_transcription
            
        except Exception as e:
            log(f"Error transcribing {audio_path}: {e}", "error")
            raise
    
    def save_transcription(self, transcription: Dict[str, Any], output_path: str):
        """
        Save transcription results to JSON file.
        
        Args:
            transcription: Transcription results
            output_path: Path to save the JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2, ensure_ascii=False)
            
            log(f"Transcription saved to: {output_path}", "info")
            
        except Exception as e:
            log(f"Error saving transcription: {e}", "error")
            raise
    
    def export_to_json_file(self, transcription_results: Dict[str, Any], json_file_path: str):
        """Export transcription results to a specific JSON file."""
        try:
            # Ensure the directory exists
            json_file = Path(json_file_path)
            json_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare the export data with metadata
            export_data = {
                "export_metadata": {
                    "total_files_processed": len(transcription_results),
                    "export_timestamp": datetime.datetime.now().isoformat(),
                    "exported_from": "Audio Transcription System",
                    "dry_run_mode": {
                        "enabled": DRY_RUN_MODE,
                        "duration_seconds": DRY_RUN_DURATION if DRY_RUN_MODE else None,
                        "status": f"DRY RUN - Only first {DRY_RUN_DURATION}s transcribed" if DRY_RUN_MODE else "FULL TRANSCRIPTION - Complete audio processed",
                        "source": "notion_database",
                        "database_id": NOTION_CONFIG_DATABASE_ID
                    },
                    "models_used": {
                        "faster_whisper": self.faster_whisper_model,
                        "whisperx_alignment": "en"
                    }
                },
                "transcriptions": transcription_results
            }
            
            # Save to the specified JSON file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Create and save the readable text report
            readable_report = create_readable_transcription_report(export_data)
            text_file_path = json_file_path.replace('.json', '.txt')
            save_text_file(text_file_path, readable_report)
            
            log(f"📄 JSON export saved to: {json_file_path}", "success")
            log(f"📄 Readable report saved to: {text_file_path}", "success")
            
        except Exception as e:
            log(f"Error exporting to JSON file: {e}", "error")
            raise
    
    def transcribe_folder(self, audio_folder: str, output_folder: str = None, channel_name: str = None) -> Dict[str, Any]:
        """
        Transcribe all audio files in a folder.
        
        Args:
            audio_folder: Path to folder containing audio files
            output_folder: Path to save transcription results (optional)
            channel_name: Name of the channel for keyword loading (optional)
            
        Returns:
            Dictionary containing all transcription results
        """
        if output_folder is None:
            output_folder = os.path.join(audio_folder, "transcriptions")
        
        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = self.find_audio_files(audio_folder)
        
        if not audio_files:
            log("No audio files found in the specified folder", "warn")
            return {}
        
        # Load models if not already loaded
        if self.whisper_model is None:
            self.load_models()
        
        all_results = {}
        
        for i, audio_file in enumerate(audio_files, 1):
            log(f"Processing file {i}/{len(audio_files)}: {audio_file.name}", "info")
            
            try:
                # Transcribe the audio file with channel keywords
                result = self.transcribe_audio_file(str(audio_file), channel_name)
                
                # Save individual transcription
                output_filename = f"{audio_file.stem}_transcription.json"
                output_path = os.path.join(output_folder, output_filename)
                self.save_transcription(result, output_path)
                
                # Store in results dictionary
                all_results[str(audio_file)] = result
                
            except Exception as e:
                log(f"Failed to transcribe {audio_file}: {e}", "error")
                all_results[str(audio_file)] = {"error": str(e)}
        
        # Save combined results
        combined_output_path = os.path.join(output_folder, "all_transcriptions.json")
        self.save_transcription(all_results, combined_output_path)
        
        log(f"Batch transcription completed. Results saved to: {output_folder}", "info")
        return all_results
    
    def fetch_notion_script(self, page_id: str) -> str:
        """Fetch the actual script content from a Notion page."""
        if not self.notion_client:
            return ""
        
        try:
            # Get page content
            page_content = self.notion_client.blocks.children.list(block_id=page_id)
            
            script_text = ""
            for block in page_content.get("results", []):
                if block["type"] == "paragraph":
                    rich_text = block.get("paragraph", {}).get("rich_text", [])
                    for text_obj in rich_text:
                        script_text += text_obj.get("text", {}).get("content", "")
                elif block["type"] == "heading_1":
                    rich_text = block.get("heading_1", {}).get("rich_text", [])
                    for text_obj in rich_text:
                        script_text += text_obj.get("text", {}).get("content", "")
                elif block["type"] == "heading_2":
                    rich_text = block.get("heading_2", {}).get("rich_text", [])
                    for text_obj in rich_text:
                        script_text += text_obj.get("text", {}).get("content", "")
                elif block["type"] == "heading_3":
                    rich_text = block.get("heading_3", {}).get("rich_text", [])
                    for text_obj in rich_text:
                        script_text += text_obj.get("text", {}).get("content", "")
                elif block["type"] == "bulleted_list_item":
                    rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                    for text_obj in rich_text:
                        script_text += text_obj.get("text", {}).get("content", "")
                elif block["type"] == "numbered_list_item":
                    rich_text = block.get("numbered_list_item", {}).get("rich_text", [])
                    for text_obj in rich_text:
                        script_text += text_obj.get("text", {}).get("content", "")
                
                script_text += " "  # Add space between blocks
            
            # Clean up the text
            script_text = re.sub(r'\s+', ' ', script_text).strip()
            return script_text
            
        except Exception as e:
            log(f"Error fetching Notion script: {e}", "error")
            return ""
    
    def align_script_with_transcription(self, script_text: str, transcription_segments: List[Dict]) -> List[Dict]:
        """
        Align script with transcription by only fixing spelling errors within each segment.
        Preserves exact transcription structure - only replaces misspelled words found in Notion script.
        """
        try:
            # Build word mapping from Notion script for spelling corrections
            notion_word_map = self._build_notion_word_mapping(script_text)
            
            aligned_segments = []
            
            for i, segment in enumerate(transcription_segments):
                try:
                    segment_text = segment["text"].strip()
                    if not segment_text:
                        aligned_segments.append(segment)
                        continue
                    
                    # Fix spelling errors in this segment only
                    corrected_text = self._fix_spelling_in_segment(segment_text, notion_word_map)
                    
                    # Create aligned segment with preserved timing
                    aligned_segment = {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": corrected_text,
                        "original_transcription": segment_text,
                        "was_corrected": corrected_text.strip() != segment_text.strip()
                    }
                    
                    # Handle word-level timestamps if available
                    if "words" in segment and isinstance(segment["words"], list):
                        aligned_segment["words"] = self._fix_word_timestamps_spelling(
                            segment["words"], 
                            segment_text, 
                            corrected_text
                        )
                    
                    aligned_segments.append(aligned_segment)
                    
                except Exception as e:
                    log(f"Error processing segment {i+1}: {e}", "error")
                    aligned_segments.append(segment)  # Keep original on error
            
            return aligned_segments
            
        except Exception as e:
            log(f"Error aligning script with transcription: {e}", "error")
            return transcription_segments
    
    def _build_notion_word_mapping(self, script_text: str) -> Dict[str, str]:
        """
        Build a simple mapping of words from Notion script for spelling correction.
        Conservative approach - only maps exact words that appear in the script.
        """
        word_mapping = {}
        
        # Extract all words from Notion script and normalize them
        words = re.findall(r'\b\w+\b', script_text)
        
        for word in words:
            # Create mapping: lowercase -> original case
            key = word.lower()
            # Always keep the version that appears in the script
            word_mapping[key] = word
        
        return word_mapping
    
    def _fix_spelling_in_segment(self, segment_text: str, notion_word_map: Dict[str, str]) -> str:
        """
        Conservative spelling correction - only fix obvious misspellings.
        """
        words = segment_text.split()
        corrected_words = []
        
        for word in words:
            # Extract core word without punctuation
            match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', word)
            if not match or not match.group(2):
                corrected_words.append(word)
                continue
                
            prefix, core_word, suffix = match.groups()
            
            # Check for direct spelling corrections
            corrected_core = self._get_spelling_correction(core_word, notion_word_map)
            
            # Reconstruct with original punctuation
            corrected_word = prefix + corrected_core + suffix
            corrected_words.append(corrected_word)
        
        return " ".join(corrected_words)
    
    def _get_spelling_correction(self, word: str, notion_word_map: Dict[str, str]) -> str:
        """
        Get spelling correction for a single word using conservative rules.
        """
        word_lower = word.lower()
        
        # Direct match in notion script - use the script version
        if word_lower in notion_word_map:
            script_version = notion_word_map[word_lower]
            
            # Only replace if it's actually a spelling difference
            if word != script_version:
                return script_version
        
        # Handle common transcription patterns for similar sounding words
        corrections = {
            "cat": "Katt",      # Cat Williams -> Katt Williams
            "parents": "Terrence",  # Parents Howard -> Terrence Howard  
            "shug": "Suge",     # Shug Knight -> Suge Knight
            "terence": "Terrence",  # Terence -> Terrence
        }
        
        if word_lower in corrections:
            # Verify the correction makes sense by checking if it's in the notion script
            correction = corrections[word_lower]
            if correction.lower() in notion_word_map:
                return correction
        
        # No correction needed
        return word
    
    def _fix_word_timestamps_spelling(self, original_words: List[Dict], 
                                    original_text: str, corrected_text: str) -> List[Dict]:
        """
        Update word-level timestamps while preserving structure after spelling fixes.
        """
        try:
            if not original_words or not corrected_text:
                return original_words
            
            original_word_list = original_text.split()
            corrected_word_list = corrected_text.split()
            
            # MUST maintain exact word count - if different, keep original
            if len(original_word_list) != len(corrected_word_list):
                return original_words
            
            corrected_words = []
            for i, word_data in enumerate(original_words):
                if not isinstance(word_data, dict) or "start" not in word_data or "end" not in word_data:
                    corrected_words.append(word_data)
                    continue
                
                if i < len(corrected_word_list):
                    original_word = word_data.get("text", "").strip()
                    corrected_word = corrected_word_list[i].strip()
                    
                    # Extract core words for comparison (remove punctuation)
                    orig_core = re.sub(r'[^\w]', '', original_word)
                    corr_core = re.sub(r'[^\w]', '', corrected_word)
                    
                    corrected_words.append({
                        "start": word_data["start"],
                        "end": word_data["end"],
                        "text": corrected_word,
                        "original_text": original_word,
                        "was_corrected": orig_core.lower() != corr_core.lower()
                    })
                else:
                    corrected_words.append(word_data)
            
            return corrected_words
            
        except Exception as e:
            log(f"Error fixing word timestamps for spelling: {e}", "error")
            return original_words


def main():
    """Main function to run the transcription process."""
    
    # ================================
    # CONFIGURATION SECTION
    # ================================
    # 🎯 AUDIO FOLDER PATH - CHANGE THIS AS NEEDED
    AUDIO_FOLDER_PATH = r"/Users/superman/Desktop/Celebrity Voice Overs"
    
    # 📄 MAIN JSON EXPORT FILE - CHANGE THIS AS NEEDED
    MAIN_JSON_EXPORT_PATH = "JSON Files/2. Transcriptions.json"
    
    # 📄 SELECTED VIDEO JSON FILE
    SELECTED_VIDEO_JSON_PATH = "JSON Files/1. Selected Video.json"
    
    # 🔑 NOTION INTEGRATION - ADD YOUR TOKEN HERE
    NOTION_TOKEN = "ntn_cC7520095381SElmcgTOADYsGnrABFn2ph1PrcaGSst2dv"  # Your Notion integration token
    
    # 🤖 MODEL CONFIGURATION - CHANGE THESE AS NEEDED
    FASTER_WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large, large-v2, large-v3
    DEVICE = "cpu"  # Options: auto, cpu, cuda - Using CPU for reliability
    COMPUTE_TYPE = "int8"  # Options: float16, int8, float32 - int8 for better CPU performance
    # ================================
    
    # Fetch configuration from Notion database
    global DRY_RUN_MODE, COMPOSITE_APPLY_INTERVAL, SPLIT_SCREEN_ENABLED
    DRY_RUN_MODE, COMPOSITE_APPLY_INTERVAL, SPLIT_SCREEN_ENABLED = fetch_configuration_from_notion(NOTION_TOKEN)
    
    log("🎤 AUDIO TRANSCRIPTION SYSTEM", "header")
    print("=" * 50)
    
    # Display configuration status from Notion
    log(f"⚙️  Configuration from Notion Database:", "info")
    if DRY_RUN_MODE:
        log(f"   🧪 DRY RUN MODE: ENABLED - Will transcribe only first {DRY_RUN_DURATION} seconds", "warn")
    else:
        log(f"   🧪 DRY RUN MODE: DISABLED - Will transcribe full audio", "info")
    log(f"   🎬 COMPOSITE CLIPS: Apply every {COMPOSITE_APPLY_INTERVAL} clips", "info")
    print("=" * 50)
    
    # Read the selected video title from JSON file
    try:
        with open(SELECTED_VIDEO_JSON_PATH, 'r', encoding='utf-8') as f:
            selected_video_data = json.load(f)
        
        target_title = selected_video_data.get("title", "")
        channel_name = selected_video_data.get("channel", "")
        page_id = selected_video_data.get("page_id", "")
        
        if not target_title:
            log("No title found in Selected Video.json", "error")
            return
        
        log(f"Target Audio: {Colors.CYAN}{target_title}{Colors.RESET}", "info")
        log(f"Channel: {Colors.YELLOW}{channel_name}{Colors.RESET}", "info")
        print("=" * 50)
        
        # Initialize transcriber and process file
        transcriber = AudioTranscriber(
            faster_whisper_model=FASTER_WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            notion_token=NOTION_TOKEN
        )
        
        # Find and process the audio file
        audio_files = transcriber.find_audio_files(AUDIO_FOLDER_PATH)
        target_audio_file = next((f for f in audio_files if f.stem == target_title), None)
        
        if target_audio_file is None:
            log(f"No audio file found with title '{target_title}'", "error")
            return
        
        # Process the file
        transcriber.load_models()
        start_time = time.time()
        result = transcriber.transcribe_audio_file(str(target_audio_file), channel_name, page_id)
        
        # Get correction statistics
        correction_stats = None
        if result.get("segments") and result["segments"][0].get("correction_stats"):
            correction_stats = result["segments"][0]["correction_stats"]
        
        # Export results
        transcriber.export_to_json_file({str(target_audio_file): result}, MAIN_JSON_EXPORT_PATH)
        
        # Final output
        processing_time = time.time() - start_time
        print("\n" + "=" * 50)
        
        if DRY_RUN_MODE:
            log(f"🧪 DRY RUN completed in {processing_time:.2f}s (transcribed {DRY_RUN_DURATION}s of audio)", "success")
        else:
            log(f"✨ Transcription completed in {processing_time:.2f}s", "success")
        
        if correction_stats:
            total_words = correction_stats["total_words"]
            corrected_words = correction_stats["corrected_words"]
            correction_rate = correction_stats["correction_rate"]
            log(f"📝 Transcript fixed through Notion ({corrected_words}/{total_words} words corrected, {correction_rate:.1f}% different)", "success")
        
        log(f"📄 Exported to: {Colors.CYAN}{MAIN_JSON_EXPORT_PATH}{Colors.RESET}", "success")
        
        # Display processing mode summary
        print("\n" + "=" * 50)
        log("📊 PROCESSING SUMMARY:", "header")
        if DRY_RUN_MODE:
            log(f"   Mode: {Colors.YELLOW}DRY RUN{Colors.RESET} - Limited transcription", "info")
            log(f"   Duration: {Colors.CYAN}{DRY_RUN_DURATION} seconds{Colors.RESET} out of full audio", "info")
            log(f"   Purpose: Testing/preview mode", "info")
        else:
            log(f"   Mode: {Colors.GREEN}FULL TRANSCRIPTION{Colors.RESET} - Complete audio", "info")
            log(f"   Duration: {Colors.CYAN}Complete audio file{Colors.RESET} processed", "info")
            log(f"   Purpose: Production mode", "info")
        log(f"   Configuration source: {Colors.MAGENTA}Notion Database{Colors.RESET}", "info")
        print("=" * 50)
        
    except Exception as e:
        log(f"Error during transcription: {e}", "error")
        return False


if __name__ == "__main__":
    main()

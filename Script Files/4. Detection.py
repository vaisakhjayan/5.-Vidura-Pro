import json
import os
import re
from typing import Dict, List, Set, Tuple
from datetime import datetime
import openai
from openai import OpenAI
import numpy as np
from typing import Optional
import pickle
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
# Transcription source configuration
# "whisper_small" - Uses Transcription Base.txt (from Whisper Small model)
# "whisperx" - Uses JSON Files/2. Transcriptions.json (from WhisperX/faster-whisper)
TRANSCRIPTION_SOURCE = "whisper_small"

# Detection Method Configuration
# "traditional" - Uses exact keyword matching from channel JSON files
# "embeddings" - Uses OpenAI embeddings for semantic matching (still filtered by channel keywords)
DETECTION_METHOD = "traditional"

# OpenAI Embeddings Configuration (only used if DETECTION_METHOD = "embeddings")
OPENAI_API_KEY = "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"
EMBEDDING_MODEL = "text-embedding-3-large"
SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score to consider a match (0-1)
BATCH_SIZE = 100  # Number of keywords to process in each batch for embeddings

class EmbeddingDetector:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.keyword_embeddings = {}
        self.initialized = False
        self.cache_dir = Path("JSON Files/embedding_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, channel_name: str) -> Path:
        """Get the path for the cached embeddings file."""
        return self.cache_dir / f"{channel_name}_embeddings.pkl"
    
    def _load_cache(self, channel_name: str) -> Dict[str, List[float]]:
        """Load cached embeddings if they exist."""
        cache_path = self._get_cache_path(channel_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}
    
    def _save_cache(self, channel_name: str):
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(channel_name)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.keyword_embeddings, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def initialize_embeddings(self, keywords: List[str], channel_name: str):
        """Create embeddings for all keywords using batching and caching."""
        print("\nInitializing embeddings for keywords...")
        try:
            # Load cached embeddings
            cached_embeddings = self._load_cache(channel_name)
            if cached_embeddings:
                print(f"Found cached embeddings for {len(cached_embeddings)} keywords")
                self.keyword_embeddings = cached_embeddings
                
                # Find keywords that need to be processed
                keywords_to_process = [k for k in keywords if k not in cached_embeddings]
                if not keywords_to_process:
                    print("✓ All keywords found in cache")
                    self.initialized = True
                    return
                print(f"Need to process {len(keywords_to_process)} new keywords")
            else:
                keywords_to_process = keywords
            
            total_keywords = len(keywords_to_process)
            print(f"Processing {total_keywords} keywords in batches of {BATCH_SIZE}")
            
            # Process keywords in batches
            for i in range(0, total_keywords, BATCH_SIZE):
                batch = keywords_to_process[i:i + BATCH_SIZE]
                print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(total_keywords + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                try:
                    response = self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch,
                        encoding_format="float"
                    )
                    
                    # Add embeddings to our dictionary
                    for keyword, embedding_data in zip(batch, response.data):
                        self.keyword_embeddings[keyword] = embedding_data.embedding
                    
                    print(f"✓ Processed {len(batch)} keywords")
                    
                    # Save cache after each batch
                    self._save_cache(channel_name)
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print("\n✓ Embeddings creation completed")
            self.initialized = True
            
        except Exception as e:
            print(f"\nError creating embeddings: {e}")
            self.initialized = False
    
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a piece of text."""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            return None
    
    def find_semantic_matches(self, text: str, allowed_keywords: Set[str]) -> List[Tuple[str, float]]:
        """Find semantic matches in text, filtered by allowed keywords."""
        if not self.initialized or not text.strip():
            return []
        
        try:
            # Get embedding for the text
            text_embedding = self.get_text_embedding(text)
            if not text_embedding:
                return []
            
            # Calculate similarities with all keyword embeddings
            matches = []
            for keyword, keyword_embedding in self.keyword_embeddings.items():
                # Only check keywords that are in our allowed set
                if keyword in allowed_keywords:
                    similarity = self.calculate_similarity(text_embedding, keyword_embedding)
                    if similarity >= SIMILARITY_THRESHOLD:
                        matches.append((keyword, similarity))
            
            # Sort by similarity score
            return sorted(matches, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            print(f"Error finding semantic matches: {e}")
            return []
    
    @staticmethod
    def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def load_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def save_text_file(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_transcription_base_txt() -> List[Dict]:
    """Load and parse the Transcription Base.txt file into a format compatible with our detection system."""
    try:
        with open('Transcription Base.txt', 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content by timestamp markers [XX:XX:XX]
        segments = []
        lines = content.split('\n')
        current_segment = {"text": "", "start": 0.0}

        for line in lines:
            # Check for timestamp
            timestamp_match = re.match(r'\[(\d+):(\d+):(\d+)\](.*)', line)
            if timestamp_match:
                # If we have a previous segment, save it
                if current_segment["text"].strip():
                    segments.append(current_segment)

                # Parse timestamp to seconds
                hours, minutes, seconds = map(int, timestamp_match.groups()[:3])
                total_seconds = hours * 3600 + minutes * 60 + seconds
                text = timestamp_match.group(4).strip()

                # Create new segment
                current_segment = {
                    "text": text,
                    "start": float(total_seconds),
                    "end": float(total_seconds) + 5.0  # Approximate 5-second segments
                }
            elif line.strip():
                # Append non-empty lines to current segment
                current_segment["text"] += " " + line.strip()

        # Add the last segment if it exists
        if current_segment["text"].strip():
            segments.append(current_segment)

        return segments
    except Exception as e:
        print(f"Error loading Transcription Base.txt: {e}")
        return []

def get_channel_keywords(channel_name: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    # Load channel keywords
    channel_file_path = os.path.join('Channel JSON Files', f'{channel_name}.json')
    channel_data = load_json_file(channel_file_path)
    
    # Get keywords grouped by their main keyword
    main_keywords = {}
    words_keywords = {}
    
    # Get keywords from the channel-specific section
    channel_section = channel_data[channel_name]
    for main_keyword, variations in channel_section.items():
        if isinstance(variations, list):
            # Sort variations by length (longest first) to prioritize longer matches
            main_keywords[main_keyword] = sorted(variations, key=len, reverse=True)
    
    # Get keywords from the Words section if it exists
    if "Words" in channel_data:
        for main_keyword, variations in channel_data["Words"].items():
            if isinstance(variations, list):
                # Sort variations by length (longest first) to prioritize longer matches
                words_keywords[main_keyword] = sorted(variations, key=len, reverse=True)
    
    return main_keywords, words_keywords

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}:{remaining_seconds:05.2f}"
    return f"{remaining_seconds:.2f}"

def find_word_timestamp(words: List[dict], target_word: str, all_variations: List[str] = None) -> Tuple[float, str]:
    """
    Find the timestamp where a celebrity name/keyword is mentioned using word-level timing.
    Returns tuple of (timestamp, matched_text) or (None, None) if not found.
    """
    if not words:
        return None, None
    
    # If all_variations provided, use them; otherwise use just the target_word
    search_variations = all_variations if all_variations else [target_word]
    
    # Convert all variations to lowercase for case-insensitive matching
    search_variations_lower = [v.lower() for v in search_variations]
    
    # Find the first occurrence among all variations
    earliest_timestamp = None
    matched_variation = None
    
    for word_data in words:
        if 'text' in word_data and 'start' in word_data:
            word_text = word_data['text'].lower()
            
            # Check if this word matches any of the search variations
            if word_text in search_variations_lower:
                word_timestamp = word_data['start']
                # Find which variation matched
                matched_idx = search_variations_lower.index(word_text)
                current_matched_variation = search_variations[matched_idx]
                
                if earliest_timestamp is None or word_timestamp < earliest_timestamp:
                    earliest_timestamp = word_timestamp
                    matched_variation = current_matched_variation
    
    return earliest_timestamp, matched_variation

def create_readable_detections_report(detections_data: dict) -> str:
    """Create a readable text format of the detections data"""
    video = detections_data['video']
    detections = detections_data['detections']
    
    # Group detections by keyword (needed for statistics and summary)
    grouped_detections = {}
    for detection in detections:
        keyword = detection['keyword']
        if keyword not in grouped_detections:
            grouped_detections[keyword] = []
        grouped_detections[keyword].append(detection)
    
    # Timeline section (actual order)
    sorted_detections = sorted(detections, key=lambda d: d['timestamp'])
    report = []
    report.append("=" * 80)
    report.append("CELEBRITY KEYWORD DETECTIONS SUMMARY")
    report.append("=" * 80)
    report.append("")
    report.append(f"Video Title: {video['title']}")
    report.append(f"Channel: {video['channel']}")
    report.append(f"Page ID: {video['page_id']}")
    report.append(f"Video Selected: {video['selected_at']}")
    report.append(f"Detection Generated: {detections_data['timestamp']}")
    report.append("")
    report.append(f"Total Detections: {len(detections)}")
    report.append("")
    report.append("-" * 80)
    report.append("DETECTION TIMELINE (Actual Order)")
    report.append("-" * 80)
    report.append("")
    for i, detection in enumerate(sorted_detections, 1):
        keyword = detection['keyword']
        timestamp = detection['timestamp']
        matched_text = detection['matched_text']
        detection_type = detection['type']
        formatted_timestamp = detection['formatted_timestamp']
        # Determine match type
        if matched_text.lower() == keyword.lower():
            match_type = "Exact Match"
        else:
            match_type = "Partial Match"
        report.append(f"   {i}. 📍 {formatted_timestamp} ({timestamp}s) - {keyword.upper()} - {match_type}: \"{matched_text}\"")
        report.append(f"      Type: {detection_type}")
    report.append("")
    
    report.append("-" * 80)
    report.append("DETECTION STATISTICS")
    report.append("-" * 80)
    report.append("")
    
    # Statistics section
    for keyword, keyword_detections in grouped_detections.items():
        report.append(f"{keyword}:")
        report.append(f"  • Total Detections: {len(keyword_detections)}")
        
        # Determine match type
        first_detection = keyword_detections[0]
        if first_detection['matched_text'].lower() == keyword.lower():
            match_type = "Exact"
        else:
            match_type = f"Partial (\"{first_detection['matched_text']}\")"
        report.append(f"  • Match Type: {match_type}")
        
        # First appearance
        first_timestamp = min(d['timestamp'] for d in keyword_detections)
        report.append(f"  • First Appearance: {format_timestamp(first_timestamp)}")
        
        # If multiple detections, add last appearance and span
        if len(keyword_detections) > 1:
            last_timestamp = max(d['timestamp'] for d in keyword_detections)
            span = last_timestamp - first_timestamp
            report.append(f"  • Last Appearance: {format_timestamp(last_timestamp)}")
            report.append(f"  • Detection Span: {span:.2f} seconds")
        
        report.append("")
    
    # Overall statistics
    if detections:
        total_window = max(d['timestamp'] for d in detections)
        detection_frequency = total_window / len(detections) if len(detections) > 0 else 0
        
        report.append(f"Total Detection Window: {total_window:.2f} seconds")
        report.append(f"Detection Frequency: 1 every {detection_frequency:.2f} seconds")
        report.append("")
    
    report.append("-" * 80)
    report.append("DETECTION SUMMARY")
    report.append("-" * 80)
    report.append("")
    
    # Summary section
    unique_celebrities = len(grouped_detections)
    total_matches = len(detections)
    
    # Check for exact vs partial matches
    exact_matches = sum(1 for d in detections if d['matched_text'].lower() == d['keyword'].lower())
    partial_matches = total_matches - exact_matches
    
    report.append(f"✅ {unique_celebrities} unique celebrities detected")
    report.append(f"✅ {total_matches} total keyword matches found")
    
    if exact_matches > 0 and partial_matches > 0:
        report.append("✅ Mix of exact and partial matches")
    elif exact_matches > 0:
        report.append("✅ All exact matches")
    else:
        report.append("✅ All partial matches")
    
    if detections:
        max_timestamp = max(d['timestamp'] for d in detections)
        report.append(f"✅ Detections span {max_timestamp:.2f} seconds of content")
    
    report.append("")
    report.append("=" * 80)
    report.append("End of Detection Summary")
    report.append("=" * 80)
    
    return "\n".join(report)

def detect_keywords():
    # Load selected video info
    selected_video = load_json_file('JSON Files/1. Selected Video.json')
    channel_name = selected_video['channel']
    
    print(f"\nProcessing for channel: {channel_name}")
    
    # Get keywords grouped by their main keyword and words section
    main_keywords, words_keywords = get_channel_keywords(channel_name)
    
    # Debug info for keywords
    print("\nLoaded keywords from channel JSON:")
    print(f"Main keywords: {len(main_keywords)} categories")
    print(f"Words keywords: {len(words_keywords)} categories")
    
    # Initialize embedding detector if using embeddings
    embedding_detector = None
    if DETECTION_METHOD == "embeddings":
        print("\nUsing OpenAI Embeddings for semantic detection...")
        embedding_detector = EmbeddingDetector(OPENAI_API_KEY)
        
        # Create set of all allowed keywords (main keywords and their variations)
        all_keywords = set()
        for keyword, variations in main_keywords.items():
            all_keywords.add(keyword)
            all_keywords.update(variations)
        for keyword, variations in words_keywords.items():
            all_keywords.add(keyword)
            all_keywords.update(variations)
        
        print(f"\nTotal unique keywords to process: {len(all_keywords)}")
        print("Sample of keywords:", list(all_keywords)[:5])
        
        # Initialize embeddings for all keywords
        embedding_detector.initialize_embeddings(list(all_keywords), channel_name)
    else:
        print("\nUsing traditional keyword-based detection...")
    
    # Load transcriptions based on selected source
    if TRANSCRIPTION_SOURCE == "whisper_small":
        print("Using Whisper Small transcription from Transcription Base.txt")
        segments = load_transcription_base_txt()
        transcriptions = {
            "transcriptions": {
                "Transcription Base.txt": {
                    "segments": segments
                }
            }
        }
    else:  # whisperx
        print("Using WhisperX transcription from JSON Files/2. Transcriptions.json")
        transcriptions = load_json_file('JSON Files/2. Transcriptions.json')
    
    # Initialize detections list
    detections = []
    
    # Process each segment
    file_path = next(iter(transcriptions['transcriptions']))
    file_data = transcriptions['transcriptions'][file_path]
    
    # Keep track of detected keywords to avoid duplicates
    detected_keywords = set()
    
    for segment in file_data['segments']:
        if 'text' not in segment:
            continue
            
        text = segment['text'].lower()
        segment_timestamp = segment['start']
        
        if DETECTION_METHOD == "embeddings":
            # Semantic detection using embeddings
            allowed_keywords = set()
            for keyword, variations in main_keywords.items():
                allowed_keywords.add(keyword)
                allowed_keywords.update(variations)
            
            semantic_matches = embedding_detector.find_semantic_matches(text, allowed_keywords)
            
            for keyword, similarity in semantic_matches:
                detection_key = f"{keyword}_{segment_timestamp}"
                
                if detection_key not in detected_keywords:
                    detected_keywords.add(detection_key)
                    detections.append({
                        "type": "semantic",
                        "keyword": keyword,
                        "matched_text": text,  # Store full context
                        "timestamp": segment_timestamp,
                        "formatted_timestamp": format_timestamp(segment_timestamp),
                        "similarity_score": similarity,
                        "segment_start": segment_timestamp
                    })
        
        else:  # Traditional keyword-based detection
            # Check main keywords
            for main_keyword, variations in main_keywords.items():
                for variation in variations:
                    variation_lower = variation.lower()
                    pattern = r'\b' + re.escape(variation_lower) + r'\b'
                    if re.search(pattern, text):
                        timestamp = None
                        matched_text = variation
                        
                        if 'words' in segment:
                            timestamp, word_matched_text = find_word_timestamp(segment['words'], variation, variations)
                            if word_matched_text:
                                matched_text = word_matched_text
                        
                        if timestamp is None:
                            timestamp = segment_timestamp
                        
                        detection_key = f"{main_keyword}_{timestamp}"
                        if detection_key not in detected_keywords:
                            detected_keywords.add(detection_key)
                            detections.append({
                                "type": "main",
                                "keyword": main_keyword,
                                "matched_text": matched_text,
                                "timestamp": timestamp,
                                "formatted_timestamp": format_timestamp(timestamp),
                                "segment_start": segment_timestamp,
                                "word_level_detection": timestamp != segment_timestamp
                            })
                        break
            
            # Check words keywords
            for word_keyword, variations in words_keywords.items():
                for variation in variations:
                    variation_lower = variation.lower()
                    pattern = r'\b' + re.escape(variation_lower) + r'\b'
                    if re.search(pattern, text):
                        timestamp = None
                        matched_text = variation
                        
                        if 'words' in segment:
                            timestamp, word_matched_text = find_word_timestamp(segment['words'], variation, variations)
                            if word_matched_text:
                                matched_text = word_matched_text
                        
                        if timestamp is None:
                            timestamp = segment_timestamp
                        
                        detection_key = f"{word_keyword}_{timestamp}"
                        if detection_key not in detected_keywords:
                            detected_keywords.add(detection_key)
                            detections.append({
                                "type": "words",
                                "keyword": word_keyword,
                                "matched_text": matched_text,
                                "timestamp": timestamp,
                                "formatted_timestamp": format_timestamp(timestamp),
                                "segment_start": segment_timestamp,
                                "word_level_detection": timestamp != segment_timestamp
                            })
                        break
    
    # Create the final detections data structure
    detections_data = {
        "video": selected_video,
        "detections": detections,
        "timestamp": datetime.now().isoformat(),
        "transcription_source": TRANSCRIPTION_SOURCE,
        "detection_method": {
            "type": DETECTION_METHOD,
            "embedding_model": EMBEDDING_MODEL if DETECTION_METHOD == "embeddings" else None,
            "similarity_threshold": SIMILARITY_THRESHOLD if DETECTION_METHOD == "embeddings" else None
        }
    }
    
    # Save detections to JSON file
    save_json_file('JSON Files/3. Detections.json', detections_data)
    
    # Create and save the readable text report
    readable_report = create_readable_detections_report(detections_data)
    save_text_file('JSON Files/3. Detections.txt', readable_report)
    
    # Print detections for feedback
    print(f"\nUsing transcription source: {TRANSCRIPTION_SOURCE}")
    print(f"Detection method: {DETECTION_METHOD.upper()}")
    
    for detection in detections:
        if detection["type"] == "semantic":
            print(f"[Semantic] Keyword '{detection['keyword']}' found at {detection['formatted_timestamp']} (similarity: {detection['similarity_score']:.2f})")
        else:
            prefix = "[Words]" if detection["type"] == "words" else ""
            word_level_info = ""
            if "word_level_detection" in detection and detection["word_level_detection"]:
                word_level_info = f" [WORD-LEVEL: {detection['formatted_timestamp']} vs SEGMENT: {format_timestamp(detection['segment_start'])}]"
            elif "segment_start" in detection:
                word_level_info = f" [SEGMENT-LEVEL: {detection['formatted_timestamp']}]"
            
            print(f"{prefix}Keyword '{detection['keyword']}' (matched '{detection['matched_text']}') found at {detection['formatted_timestamp']} seconds{word_level_info}")
    
    print(f"\nDetections saved to JSON Files/3. Detections.json")
    print(f"Readable report saved to JSON Files/3. Detections.txt")

if __name__ == "__main__":
    try:
        detect_keywords()
    except Exception as e:
        print(f"Error: {str(e)}")

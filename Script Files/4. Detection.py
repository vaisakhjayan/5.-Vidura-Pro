import json
import os
import re
from typing import Dict, List, Set, Tuple
from datetime import datetime

def load_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(file_path: str, data: dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def save_text_file(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

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
    
    # Get keywords grouped by their main keyword and words section
    main_keywords, words_keywords = get_channel_keywords(channel_name)
    
    # Load transcriptions
    transcriptions = load_json_file('JSON Files/2. Transcriptions.json')
    
    # Initialize detections list
    detections = []
    
    # Process each segment
    # The transcriptions dict has a single key with the file path
    file_path = next(iter(transcriptions['transcriptions']))
    file_data = transcriptions['transcriptions'][file_path]
    
    # Keep track of detected keywords to avoid duplicates
    detected_keywords = set()
    
    for segment in file_data['segments']:
        # Skip segments that don't have text
        if 'text' not in segment:
            continue
            
        text = segment['text'].lower()
        
        # First check main keywords
        for main_keyword, variations in main_keywords.items():
            # Check variations in order (longest first)
            for variation in variations:
                variation_lower = variation.lower()
                # Use word boundaries to ensure we match complete words only
                # This prevents matching "Al" in "real", "also", etc.
                pattern = r'\b' + re.escape(variation_lower) + r'\b'
                if re.search(pattern, text):
                    # If we have word-level timing, use it for more accurate timestamps
                    timestamp = None
                    matched_text = variation  # Default to the variation that matched in text
                    if 'words' in segment:
                        timestamp, word_matched_text = find_word_timestamp(segment['words'], variation, variations)
                        if word_matched_text:  # If word-level detection found a match, use it
                            matched_text = word_matched_text
                        # Debug for first Diddy issue
                        if main_keyword == "Diddy" and segment['start'] < 17:
                            print(f"DEBUG: First Diddy segment - variation='{variation}', all_variations={variations}")
                            print(f"DEBUG: Word result - timestamp={timestamp}, matched_text='{word_matched_text}'")
                            diddy_words = [w for w in segment['words'] if 'diddy' in w.get('text', '').lower()]
                            print(f"DEBUG: Available Diddy words: {diddy_words}")
                    
                    # Fall back to segment timing if word timing is not available
                    segment_timestamp = segment['start']
                    if timestamp is None:
                        timestamp = segment_timestamp
                    
                    # Create a unique key for this detection to avoid duplicates
                    detection_key = f"{main_keyword}_{timestamp}"
                    if detection_key not in detected_keywords:
                        detected_keywords.add(detection_key)
                        detections.append({
                            "type": "main",
                            "keyword": main_keyword,
                            "matched_text": matched_text,
                            "timestamp": timestamp,
                            "formatted_timestamp": format_timestamp(timestamp),
                            "segment_start": segment_timestamp,  # Keep segment info for reference
                            "word_level_detection": timestamp != segment_timestamp  # Flag if word-level was used
                        })
                    # Once we find a match in a keyword group, move to next group
                    break
        
        # Then check words keywords
        for word_keyword, variations in words_keywords.items():
            # Check variations in order (longest first)
            for variation in variations:
                variation_lower = variation.lower()
                # Use word boundaries to ensure we match complete words only
                # This prevents matching "Al" in "real", "also", etc.
                pattern = r'\b' + re.escape(variation_lower) + r'\b'
                if re.search(pattern, text):
                    # If we have word-level timing, use it for more accurate timestamps
                    timestamp = None
                    matched_text = variation  # Default to the variation that matched in text
                    if 'words' in segment:
                        timestamp, word_matched_text = find_word_timestamp(segment['words'], variation, variations)
                        if word_matched_text:  # If word-level detection found a match, use it
                            matched_text = word_matched_text
                    
                    # Fall back to segment timing if word timing is not available
                    segment_timestamp = segment['start']
                    if timestamp is None:
                        timestamp = segment_timestamp
                    
                    # Create a unique key for this detection to avoid duplicates
                    detection_key = f"{word_keyword}_{timestamp}"
                    if detection_key not in detected_keywords:
                        detected_keywords.add(detection_key)
                        detections.append({
                            "type": "words",
                            "keyword": word_keyword,
                            "matched_text": matched_text,
                            "timestamp": timestamp,
                            "formatted_timestamp": format_timestamp(timestamp),
                            "segment_start": segment_timestamp,  # Keep segment info for reference
                            "word_level_detection": timestamp != segment_timestamp  # Flag if word-level was used
                        })
                    # Once we find a match in a keyword group, move to next group
                    break
    
    # Create the final detections data structure
    detections_data = {
        "video": selected_video,
        "detections": detections,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save detections to JSON file
    save_json_file('JSON Files/3. Detections.json', detections_data)
    
    # Create and save the readable text report
    readable_report = create_readable_detections_report(detections_data)
    save_text_file('JSON Files/3. Detections.txt', readable_report)
    
    # Print detections for feedback
    for detection in detections:
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

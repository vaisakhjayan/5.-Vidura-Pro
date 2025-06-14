import openai
import json
import numpy as np
from typing import List, Dict, Tuple
import os
import re

# Set up OpenAI API key
# Note: For security, consider using environment variables instead of hardcoding
openai.api_key = "sk-QpcXPtd6-aW4Bq2O5oNpWMUEL2ELDAoQPVUEfagGrbT3BlbkFJEDecQyTi9cCi6L0b-tOQnSSiThRWYtJ8P1BBtJpXAA"

# Alternative secure method (uncomment and use environment variable):
# openai.api_key = os.getenv("OPENAI_API_KEY")

class VideoClipMatcher:
    """
    Intelligent video clip suggestion system using OpenAI embeddings.
    """
    
    def __init__(self, model="text-embedding-3-large"):
        self.model = model
        self.client = openai.OpenAI(api_key=openai.api_key)
        
    def read_selected_video(self, file_path: str = "JSON Files/1. Selected Video.json") -> Dict:
        """Read the selected video metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'channel': data.get('channel', ''),
                    'title': data.get('title', '')
                }
        except Exception as e:
            print(f"Error reading selected video file: {e}")
            return {}
    
    def read_transcription(self, file_path: str = "JSON Files/2. Transcriptions.json") -> List[Dict]:
        """Read and extract transcription segments."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract segments from transcription
            transcriptions = data.get('transcriptions', {})
            segments = []
            
            for file_key, file_data in transcriptions.items():
                file_segments = file_data.get('segments', [])
                segments.extend(file_segments)
            
            return segments
        except Exception as e:
            print(f"Error reading transcription file: {e}")
            return []
    
    def read_channel_clips(self, channel_name: str) -> Dict:
        """Read the channel's intelligent clip data."""
        try:
            file_path = f"Channel JSON Files/{channel_name}.json"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # The "Intelligent Clip" section is at the top level, not nested under channel name
            intelligent_clips = data.get('Intelligent Clip', {})
            
            return intelligent_clips
        except Exception as e:
            print(f"Error reading channel clips file: {e}")
            return {}
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0
            
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def create_transcription_segments(self, segments: List[Dict], segment_length: int = 3) -> List[Dict]:
        """Group transcription segments into larger chunks for better context."""
        grouped_segments = []
        
        for i in range(0, len(segments), segment_length):
            chunk = segments[i:i + segment_length]
            
            # Combine text from multiple segments
            combined_text = " ".join([seg.get('text', '') for seg in chunk])
            start_time = chunk[0].get('start', 0)
            end_time = chunk[-1].get('end', 0)
            
            grouped_segments.append({
                'text': combined_text.strip(),
                'start': start_time,
                'end': end_time,
                'timestamp': f"{start_time:.1f}s - {end_time:.1f}s"
            })
        
        return grouped_segments
    
    def prepare_clip_descriptions(self, intelligent_clips: Dict) -> List[Dict]:
        """Prepare clip descriptions with combined text for embedding."""
        clip_data = []
        
        for clip_name, clip_info in intelligent_clips.items():
            description = clip_info.get('description', '')
            keywords = clip_info.get('keywords', [])
            folder = clip_info.get('folder', clip_name)
            
            # Combine description and keywords for better matching
            combined_text = f"{description}. Keywords: {', '.join(keywords)}"
            
            clip_data.append({
                'name': clip_name,
                'folder': folder,
                'description': description,
                'keywords': keywords,
                'combined_text': combined_text
            })
        
        return clip_data
    
    def find_best_matches(self, transcription_segments: List[Dict], clip_data: List[Dict], 
                         similarity_threshold: float = 0.55) -> List[Tuple]:
        """Find the best clip matches for each transcription segment."""
        matches = []
        
        print("Processing transcription segments and finding matches...")
        
        # Get embeddings for all clips first
        print("Getting embeddings for video clips...")
        clip_embeddings = []
        for clip in clip_data:
            embedding = self.get_embedding(clip['combined_text'])
            clip_embeddings.append(embedding)
        
        # Process each transcription segment
        for i, segment in enumerate(transcription_segments):
            print(f"Processing segment {i+1}/{len(transcription_segments)}: {segment['timestamp']}")
            
            segment_embedding = self.get_embedding(segment['text'])
            if not segment_embedding:
                continue
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with all clips
            for j, clip_embedding in enumerate(clip_embeddings):
                if not clip_embedding:
                    continue
                    
                similarity = self.cosine_similarity(segment_embedding, clip_embedding)
                
                # Additional keyword-based filtering for better accuracy
                keyword_bonus = self._calculate_keyword_bonus(segment['text'], clip_data[j])
                adjusted_similarity = similarity + keyword_bonus
                
                if adjusted_similarity > best_similarity and adjusted_similarity >= similarity_threshold:
                    # Check for context conflicts (e.g., don't suggest funeral clips for wedding content)
                    if not self._has_context_conflict(segment['text'], clip_data[j]):
                        best_similarity = adjusted_similarity
                        best_match = clip_data[j]
            
            # Only add if we found a good match
            if best_match and best_similarity >= similarity_threshold:
                matches.append({
                    'segment': segment,
                    'clip': best_match,
                    'similarity': best_similarity
                })
        
        return matches
    
    def _calculate_keyword_bonus(self, segment_text: str, clip_data: Dict) -> float:
        """Calculate bonus score based on keyword matches."""
        bonus = 0.0
        segment_lower = segment_text.lower()
        
        # Check for direct keyword matches
        keywords = clip_data.get('keywords', [])
        for keyword in keywords:
            if keyword.lower() in segment_lower:
                bonus += 0.05  # Reduced base bonus for general keywords
        
        # Check for important specific matches with stronger bonuses
        clip_name = clip_data.get('name', '').lower()
        
        # Wedding specific bonuses - HIGHEST PRIORITY
        if 'wedding' in clip_name:
            # Very explicit wedding references get huge bonus
            if any(phrase in segment_lower for phrase in ['walked down the aisle', 'ceremony watched by millions', 'wedding day', 'their vows']):
                bonus += 0.25
            elif any(word in segment_lower for word in ['wedding', 'married', 'vows', 'ceremony', 'aisle']):
                bonus += 0.20
                
        # Funeral specific bonuses  
        if 'funeral' in clip_name:
            if any(phrase in segment_lower for phrase in ['queen elizabeth\'s funeral', 'funeral', 'elizabeth\'s funeral']):
                bonus += 0.25
            elif any(word in segment_lower for word in ['funeral', 'elizabeth', 'queen']):
                bonus += 0.20
                
        # Kids/family specific bonuses
        if 'kids' in clip_name or 'family' in clip_name:
            if any(phrase in segment_lower for phrase in ['time spent with their children', 'loving parents']):
                bonus += 0.25
            elif any(word in segment_lower for word in ['children', 'kids', 'family', 'parents']):
                bonus += 0.20
                
        # Speech specific bonuses
        if 'speech' in clip_name:
            if any(phrase in segment_lower for phrase in ['giving heartfelt speeches', 'giving speech']):
                bonus += 0.25
            elif any(word in segment_lower for word in ['speech', 'speaking', 'inspire', 'advocate']):
                bonus += 0.20
                
        # Emotional clips bonuses
        if 'emotional' in clip_name and 'sad' in clip_name:
            if any(phrase in segment_lower for phrase in ['emotional interviews', 'teary-eyed', 'shared her struggles']):
                bonus += 0.25
            elif any(word in segment_lower for word in ['emotional', 'tears', 'struggles', 'vulnerable']):
                bonus += 0.20
                
        # Suits/Acting clips - should have LOWER priority unless very explicit
        if 'suits' in clip_name:
            if any(phrase in segment_lower for phrase in ['acting in suits', 'actress', 'early days as an actress']):
                bonus += 0.15  # Lower bonus for acting references
            # Don't give bonus just for general mentions
                
        # Oprah interview specific
        if 'oprah' in clip_name:
            if any(word in segment_lower for word in ['oprah', 'interview']):
                bonus += 0.20
        
        return min(bonus, 0.3)  # Increased cap to 0.3 for very explicit matches
    
    def _has_context_conflict(self, segment_text: str, clip_data: Dict) -> bool:
        """Check if there's a context conflict between segment and clip."""
        segment_lower = segment_text.lower()
        clip_name = clip_data.get('name', '').lower()
        
        # Don't suggest funeral clips for wedding content
        if 'funeral' in clip_name and any(word in segment_lower for word in ['wedding', 'married', 'vows', 'ceremony']):
            return True
            
        # Don't suggest wedding clips for funeral content
        if 'wedding' in clip_name and any(word in segment_lower for word in ['funeral', 'death', 'mourning']):
            return True
            
        # Don't suggest emotional/sad clips for positive content
        if 'emotional' in clip_name and 'sad' in clip_name:
            if any(word in segment_lower for word in ['inspire', 'powerful', 'successful', 'happy']):
                return True
        
        return False
    
    def format_output(self, matches: List[Dict], video_info: Dict) -> str:
        """Format the output for the text file."""
        output_lines = []
        
        # Header
        output_lines.append("=" * 80)
        output_lines.append("INTELLIGENT VIDEO CLIP SUGGESTIONS")
        output_lines.append("=" * 80)
        output_lines.append(f"Channel: {video_info.get('channel', 'Unknown')}")
        output_lines.append(f"Video: {video_info.get('title', 'Unknown')}")
        output_lines.append(f"Total Matches Found: {len(matches)}")
        output_lines.append("=" * 80)
        output_lines.append("")
        
        # Matches
        for i, match in enumerate(matches, 1):
            segment = match['segment']
            clip = match['clip']
            similarity = match['similarity']
            
            # Calculate 10-second placement window (two 5-second clips)
            start_time = segment['start']
            end_time = start_time + 10.0  # 10 seconds total placement
            placement_timestamp = f"{start_time:.1f}s - {end_time:.1f}s"
            
            output_lines.append(f"MATCH #{i}")
            output_lines.append("-" * 40)
            output_lines.append(f"Timestamp: {placement_timestamp}")
            output_lines.append(f"Transcription Section:")
            output_lines.append(f'"{segment["text"]}"')
            output_lines.append("")
            output_lines.append(f"Suggested Clip: {clip['name']}")
            output_lines.append(f"Folder: {clip['folder']}")
            output_lines.append(f"Description: {clip['description']}")
            output_lines.append(f"Similarity Score: {similarity:.3f}")
            output_lines.append("")
            output_lines.append("=" * 80)
            output_lines.append("")
        
        # Summary
        if matches:
            avg_similarity = sum(match['similarity'] for match in matches) / len(matches)
            output_lines.append(f"Average Similarity Score: {avg_similarity:.3f}")
        else:
            output_lines.append("No matches found above the similarity threshold.")
        
        return "\n".join(output_lines)
    
    def save_output(self, content: str, file_path: str = "3 Large Output.txt"):
        """Save the output to a text file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Output saved to {file_path}")
        except Exception as e:
            print(f"Error saving output file: {e}")
    
    def process_video(self):
        """Main processing function."""
        print("Starting Intelligent Video Clip Matching...")
        print("=" * 60)
        
        # Step 1: Read video metadata
        print("Step 1: Reading video metadata...")
        video_info = self.read_selected_video()
        if not video_info.get('channel'):
            print("Error: Could not read video metadata")
            return
        
        print(f"Channel: {video_info['channel']}")
        print(f"Video: {video_info['title']}")
        print()
        
        # Step 2: Read transcription
        print("Step 2: Reading transcription...")
        segments = self.read_transcription()
        if not segments:
            print("Error: Could not read transcription")
            return
        
        print(f"Found {len(segments)} transcription segments")
        print()
        
        # Step 3: Read channel clips
        print("Step 3: Reading channel clip data...")
        intelligent_clips = self.read_channel_clips(video_info['channel'])
        if not intelligent_clips:
            print("Error: Could not read channel clips")
            return
        
        print(f"Found {len(intelligent_clips)} available clip types")
        print()
        
        # Step 4: Prepare data for processing
        print("Step 4: Preparing data for processing...")
        transcription_segments = self.create_transcription_segments(segments)
        clip_data = self.prepare_clip_descriptions(intelligent_clips)
        
        print(f"Created {len(transcription_segments)} transcription segments")
        print(f"Prepared {len(clip_data)} clip descriptions")
        print()
        
        # Step 5: Find matches using embeddings
        print("Step 5: Finding matches using AI embeddings...")
        matches = self.find_best_matches(transcription_segments, clip_data)
        print(f"Found {len(matches)} potential matches")
        print()
        
        # Step 6: Generate and save output
        print("Step 6: Generating output...")
        output_content = self.format_output(matches, video_info)
        self.save_output(output_content)
        
        print("Process completed successfully!")

def main():
    """Main function to run the video clip matcher."""
    matcher = VideoClipMatcher()
    matcher.process_video()

if __name__ == "__main__":
    main()

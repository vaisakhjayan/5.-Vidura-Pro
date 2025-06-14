import json
import subprocess
import os

def load_render_plan(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_ffmpeg_command(render_plan, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get video info
    video_info = render_plan['video_info']
    
    # Prepare output path
    output_path = os.path.join(output_folder, video_info['output_file'])
    
    # Sort segments by start time
    segments = sorted(render_plan["segments"], key=lambda x: x["start_time"])
    
    # Base command
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-hide_banner',
        '-loglevel', 'warning'
    ]

    # Add all media inputs
    for segment in segments:
        media_path = os.path.join(segment["media"]["folder"], segment["media"]["file"])
        if segment["media"]["type"] == "image":
            command.extend(['-loop', '1', '-t', str(segment["media"]["duration"]), '-i', media_path])
        else:
            command.extend(['-i', media_path])
    
    # Start building the filter complex for concatenation
    filter_complex = []
    
    # Process each input - trim videos to exact duration and scale/pad everything
    for i, segment in enumerate(segments):
        duration = segment["end_time"] - segment["start_time"]
        
        if segment["media"]["type"] == "image":
            # For images, they're already looped to the right duration
            filter_complex.append(
                f'[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,'
                f'pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30[v{i}];'
            )
        else:
            # For videos, trim to the exact duration needed
            filter_complex.append(
                f'[{i}:v]trim=duration={duration},setpts=PTS-STARTPTS,'
                f'scale=1920:1080:force_original_aspect_ratio=decrease,'
                f'pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30[v{i}];'
            )
    
    # Concatenate all the processed videos
    concat_inputs = ''.join([f'[v{i}]' for i in range(len(segments))])
    filter_complex.append(f'{concat_inputs}concat=n={len(segments)}:v=1:a=0[outv]')

    # Join all filter parts
    filter_string = ''.join(filter_complex)
    
    # Add the filter complex and output options
    command.extend([
        '-filter_complex', filter_string,
        '-map', '[outv]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-r', '30',  # Set output framerate
        output_path
    ])
    
    return command

def main():
    try:
        print("Starting video rendering process...")
        
        # Load render plan
        render_plan = load_render_plan('JSON Files/Render Plan V2.json')
        
        # Create output folder path
        output_folder = 'E:\\Ready To Be Refined'
        
        # Generate FFMPEG command
        ffmpeg_command = create_ffmpeg_command(render_plan, output_folder)
        
        print("Rendering video with concatenation...")
        print("Command:", ' '.join(ffmpeg_command))  # Print command for debugging
        
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\nSuccess! Video has been rendered to: {output_folder}")
        else:
            print("\nError during rendering:")
            print(result.stderr)
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e.filename}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in render plan file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main() 
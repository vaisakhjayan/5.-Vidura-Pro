import os
import sys
import datetime
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

def run_step(step_number, step_name, module_path):
    """Run a single step of the pipeline and handle any errors."""
    try:
        log(f"Step {step_number}: {step_name}", "header")
        print("=" * 50)
        
        # Import the module using importlib for better control
        import importlib.util
        
        # Load the module from the full path
        spec = importlib.util.spec_from_file_location(f"step_{step_number}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Call the appropriate function based on step number
        start_time = time.time()
        
        if step_number == 3:  # Detection step
            result = module.detect_keywords()
        elif step_number == 4:  # Clips Placement step
            # Get the current directory
            current_dir = os.path.dirname(os.path.abspath(module_path))
            
            # Construct absolute paths for input/output files
            detections_json = os.path.join(os.path.dirname(current_dir), "JSON Files", "3. Detections.json")
            text_output_file = os.path.join(os.path.dirname(current_dir), "JSON Files", "4. Render Plan V2.txt")
            json_output_file = os.path.join(os.path.dirname(current_dir), "JSON Files", "4. Render Plan V2.json")
            
            # Call the function with required parameters
            result = module.generate_render_plans_5s_blocks(detections_json, text_output_file, json_output_file)
        else:
            result = module.main()  # All other steps use main()
            
        duration = time.time() - start_time
        
        # Check result
        if result is False:  # Explicit failure
            log(f"Step {step_number} failed!", "error")
            return False
        
        log(f"Step {step_number} completed in {duration:.2f}s", "success")
        print("\n" + "=" * 50 + "\n")
        return True
        
    except Exception as e:
        log(f"Error in step {step_number}: {str(e)}", "error")
        return False

def run_pipeline_once():
    """Run the complete pipeline once."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the pipeline steps with actual file paths
    steps = [
        (1, "Notion Integration", os.path.join(script_dir, "2. Notion.py")),
        (2, "Audio Transcription", os.path.join(script_dir, "3. Transcription.py")),
        (3, "Keyword Detection", os.path.join(script_dir, "4. Detection.py")),
        (4, "Clips Placement", os.path.join(script_dir, "5. Clips Placement.py")),
        (5, "Effects Processing", os.path.join(script_dir, "6. Effects.py"))
    ]
    
    # Run each step in sequence
    for step_number, step_name, module_path in steps:
        if not run_step(step_number, step_name, module_path):
            if step_number == 1:
                # Step 1 failure means no videos to edit - this is normal
                log("No videos need editing right now", "success")
                return "no_videos"
            else:
                # Other step failures are actual errors
                log("Pipeline stopped due to error", "error")
                return "error"
        
        # Small delay between steps for readability
        time.sleep(0.5)
    
    log("🎉 Video processing completed successfully!", "success")
    return "success"

def main():
    """Main function to run the pipeline with continuous monitoring."""
    log("🎬 VIDEO PROCESSING PIPELINE", "header")
    log("🔄 Continuous monitoring mode - Press Ctrl+C to stop", "info")
    print("=" * 50)
    
    try:
        while True:
            print("\n" + "=" * 60)
            log(f"🔍 Checking for videos to edit...", "header")
            print("=" * 60)
            
            result = run_pipeline_once()
            
            if result == "success":
                log("✅ Video processed successfully!", "success")
                log("🔄 Continuing to monitor for new videos...", "info")
            elif result == "no_videos":
                log("✅ All videos are up to date!", "success")
                log("🔄 Will check again in 60 seconds...", "info")
            elif result == "error":
                log("❌ Pipeline error occurred", "error")
                log("🔄 Will retry in 60 seconds...", "warn")
            
            # Wait before next check
            next_check = datetime.datetime.now() + datetime.timedelta(seconds=60)
            log(f"⏰ Next check at {next_check.strftime('%H:%M:%S')}", "dim")
            
            # Countdown with dots
            for i in range(60, 0, -10):
                if i == 60:
                    log(f"💤 Waiting {i} seconds", "dim", newline=False)
                else:
                    print(".", end="", flush=True)
                time.sleep(10)
            print()  # New line after countdown
            
    except KeyboardInterrupt:
        print("\n")
        log("🛑 Pipeline monitoring stopped by user", "warn")
        log("Goodbye! ✌️", "header")
    except Exception as e:
        log(f"❌ Unexpected error: {str(e)}", "error")

if __name__ == "__main__":
    main()

import requests
import datetime
import time
import json
import os

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

# Notion configuration
NOTION_DATABASE_ID = "1a402cd2c14280909384df6c898ddcb3"
NOTION_TOKEN = "ntn_cC7520095381SElmcgTOADYsGnrABFn2ph1PrcaGSst2dv"
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

def get_pages_needing_editing():
    """Get pages where 'Ready to Be Edited' is checked but 'Video Edited' is not checked."""
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    
    try:
        response = requests.post(url, headers=NOTION_HEADERS, json={
            "filter": {
                "and": [
                    {
                        "property": "Ready to Be Edited",
                        "checkbox": {
                            "equals": True
                        }
                    },
                    {
                        "property": "Video Edited",
                        "checkbox": {
                            "equals": False
                        }
                    }
                ]
            }
        })
        response.raise_for_status()
        pages = response.json().get("results", [])
        return pages
    except Exception as e:
        log(f"Error querying database: {str(e)}", "error")
        return []

def get_page_title(page):
    """Extract title from Notion page."""
    try:
        title_property = page.get("properties", {}).get("title", {})
        if not title_property:
            # Try different possible title property names
            for prop_name, prop_data in page.get("properties", {}).items():
                if prop_data.get("type") == "title":
                    title_property = prop_data
                    break
        
        if title_property and title_property.get("title"):
            title_array = title_property["title"]
            if title_array and len(title_array) > 0:
                return title_array[0].get("text", {}).get("content", "Untitled")
        return "Untitled"
    except Exception as e:
        log(f"Error getting page title: {str(e)}", "error")
        return "Untitled"

def get_page_channel(page):
    """Extract channel from Notion page."""
    try:
        channel_property = page.get("properties", {}).get("Channel", {})
        if channel_property and channel_property.get("select"):
            return channel_property["select"].get("name", "Unknown Channel")
        return "Unknown Channel"
    except Exception as e:
        log(f"Error getting channel: {str(e)}", "error")
        return "Unknown Channel"

def display_videos_needing_editing(pages):
    """Display the list of videos that need editing in a clean format, including channel info."""
    if not pages:
        log("No videos need editing right now!", "success")
        return
    
    log(f"Found {len(pages)} video(s) ready for editing:", "success")
    print()  # Empty line for spacing
    
    for i, page in enumerate(pages, 1):
        title = get_page_title(page)
        channel = get_page_channel(page)
        # Format the output nicely
        log(f"{i}. {Colors.CYAN}{title}{Colors.RESET} {Colors.YELLOW}[{channel}]{Colors.RESET}", "info")
        print()  # Empty line between entries

def save_selected_video(page):
    """Save the selected video title to JSON file."""
    try:
        title = get_page_title(page)
        channel = get_page_channel(page)
        page_id = page.get("id", "")
        
        selected_video_data = {
            "title": title,
            "channel": channel,
            "page_id": page_id,
            "selected_at": datetime.datetime.now().isoformat()
        }
        
        # Ensure the JSON Files directory exists
        json_dir = "JSON Files"
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        
        json_file_path = os.path.join(json_dir, "1. Selected Video.json")
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(selected_video_data, f, indent=2, ensure_ascii=False)
        
        log(f"✓ Selected video saved: {Colors.CYAN}{title}{Colors.RESET} {Colors.YELLOW}[{channel}]{Colors.RESET}", "success")
        return True
        
    except Exception as e:
        log(f"Error saving selected video: {str(e)}", "error")
        return False

def main():
    """Main function to check for videos needing editing."""
    try:
        pages = get_pages_needing_editing()
        display_videos_needing_editing(pages)
        
        # If there are videos needing editing, select the first one and save it
        if pages:
            selected_page = pages[0]  # Select the first video
            save_selected_video(selected_page)
            
        return len(pages) > 0
    except Exception as e:
        log(f"Error during check: {str(e)}", "error")
        return False

if __name__ == "__main__":
    log("🎬 VIDEO EDITING MONITOR", "header")
    
    try:
        while True:
            log("", "header")  # Empty line with header formatting
            log(f"Monitoring For Videos Ready To Edit", "header")
            
            try:
                has_videos = main()
                if not has_videos:
                    log("All videos are up to date!", "success")
            except Exception as e:
                log(f"Error: {str(e)}", "error")
            
            # Wait before checking again
            next_check = datetime.datetime.now() + datetime.timedelta(seconds=60)
            log(f"Next check at {next_check.strftime('%H:%M:%S')}", "dim")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n")
        log("Video monitor stopped", "info")
        log("Goodbye! ✌️", "header")

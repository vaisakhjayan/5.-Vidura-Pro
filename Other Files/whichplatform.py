import platform

def get_platform():
    system = platform.system().lower()
    
    if system == "darwin":
        return "You are running on a Mac (macOS)"
    elif system == "windows":
        return "You are running on a PC (Windows)"
    elif system == "linux":
        return "You are running on Linux"
    else:
        return f"You are running on an unknown platform: {system}"

# Test the function
if __name__ == "__main__":
    result = get_platform()
    print(result)

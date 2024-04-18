import os
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Check if requirements.txt file exists
if os.path.isfile("requirements.txt"):
    print("requirements.txt file found.")

    # Check if all required packages are installed
    try:
        subprocess.check_call([os.sys.executable, "-m", "pip", "freeze"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
else:
    print("No requirements.txt file found.")

# Run the backend script
print("Running backend script...")
subprocess.check_call([os.sys.executable, "backend/main.py"])

# Run the Streamlit frontend script
print("Running Streamlit frontend script...")
subprocess.check_call(["streamlit", "run", "frontend/main.py"])
import subprocess
import sys
import os

def install_package(package):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")
        return False

def setup_dependencies():
    # Core requirements
    requirements = [
        'websocket-client',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib'
    ]
    
    # Optional ML requirements
    ml_requirements = [
        'tensorflow'
    ]
    
    print("Installing core dependencies...")
    for req in requirements:
        if install_package(req):
            print(f"Successfully installed {req}")
        else:
            print(f"Warning: Could not install {req}")
    
    print("\nAttempting to install machine learning dependencies...")
    for req in ml_requirements:
        if install_package(req):
            print(f"Successfully installed {req}")
        else:
            print(f"Warning: Could not install {req}. Some ML features may be unavailable.")

if __name__ == "__main__":
    setup_dependencies()

#!/usr/bin/env python3
"""
Download all CNN data directories from Modal volume.
"""

import subprocess
import os

def download_cnn_directories():
    """Download all CNN directories from the Modal volume"""
    
    # First, let's check what's in the CNN volume
    print("📋 Checking CNN volume contents...")
    try:
        result = subprocess.run([
            "modal", "volume", "ls", "cnn-csv-logs"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("Available CNN directories:")
            print(result.stdout)
            
            # Extract directory names from the output
            directories = []
            for line in result.stdout.split('\n'):
                line = line.strip()
                if 'ResNet18_' in line and 'dir' in line:
                    # Extract directory name from table format
                    parts = line.split('│')
                    if len(parts) > 1:
                        dir_name = parts[1].strip()
                        if dir_name and 'ResNet18_' in dir_name:
                            directories.append(dir_name)
            
            if not directories:
                # Fallback: assume standard naming pattern
                directories = [
                    "ResNet18_ReLU_20250911_094500",
                    "ResNet18_GELU_20250911_100000", 
                    "ResNet18_Sigmoid_20250911_105000",
                    "ResNet18_Tanh_20250911_110000",
                    "ResNet18_ELU_20250911_115000",
                    "ResNet18_LeakyReLU_20250911_120000",
                    "ResNet18_SiLU_20250911_125000",
                    "ResNet18_Mish_20250911_130000"
                ]
                print("⚠️  Using fallback directory names")
        else:
            print(f"❌ Could not list CNN volume: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking volume: {e}")
        return False
    
    # Create local directory
    os.makedirs("cnn_data", exist_ok=True)
    
    print(f"\n📥 Downloading {len(directories)} CNN data directories...")
    
    success_count = 0
    for directory in directories:
        try:
            print(f"📁 Downloading {directory}...")
            result = subprocess.run([
                "modal", "volume", "get", "cnn-csv-logs", directory, "cnn_data/"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ Downloaded {directory}")
                success_count += 1
            else:
                print(f"❌ Failed to download {directory}: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout downloading {directory}")
        except Exception as e:
            print(f"❌ Error downloading {directory}: {e}")
    
    print(f"\n✅ Downloaded {success_count}/{len(directories)} directories")
    
    if success_count > 0:
        print("\n📊 Now you can run:")
        print("   python analyze_cnn_results.py")
        
        # Show what we downloaded
        print(f"\n📁 Downloaded data:")
        if os.path.exists("cnn_data"):
            for item in os.listdir("cnn_data"):
                item_path = os.path.join("cnn_data", item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    print(f"   {item}/ ({len(files)} files)")
    else:
        print("\n❌ No directories downloaded successfully")
        print("💡 Try manually:")
        print("   modal volume ls cnn-csv-logs")
        print("   modal volume get cnn-csv-logs <directory-name> cnn_data/")
    
    return success_count > 0

if __name__ == "__main__":
    print("🚀 CNN Data Downloader")
    print("=" * 40)
    download_cnn_directories()

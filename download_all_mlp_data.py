#!/usr/bin/env python3
"""
Download all MLP data directories from Modal volume.
"""

import subprocess
import os

def download_mlp_directories():
    """Download all MLP directories from the Modal volume"""
    
    # List of directories we know exist from the modal volume ls output
    directories = [
        "MLP_ReLU_20250910_155206",
        "MLP_GELU_20250910_155500", 
        "MLP_Sigmoid_20250910_155757",
        "MLP_Tanh_20250910_160053",
        "MLP_ELU_20250910_160350",
        "MLP_LeakyReLU_20250910_160646",
        "MLP_SiLU_20250910_160941",
        "MLP_Mish_20250910_161236"
    ]
    
    # Create local directory
    os.makedirs("mlp_data", exist_ok=True)
    
    print("📥 Downloading MLP data directories...")
    
    success_count = 0
    for directory in directories:
        try:
            print(f"📁 Downloading {directory}...")
            result = subprocess.run([
                "modal", "volume", "get", "mlp-csv-logs", directory, "mlp_data/"
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
        print("   python analyze_mlp_results.py")
        
        # Show what we downloaded
        print(f"\n📁 Downloaded data:")
        if os.path.exists("mlp_data"):
            for item in os.listdir("mlp_data"):
                item_path = os.path.join("mlp_data", item)
                if os.path.isdir(item_path):
                    files = os.listdir(item_path)
                    print(f"   {item}/ ({len(files)} files)")
    else:
        print("\n❌ No directories downloaded successfully")
    
    return success_count > 0

if __name__ == "__main__":
    print("🚀 MLP Data Downloader")
    print("=" * 40)
    download_mlp_directories()

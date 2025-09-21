#!/usr/bin/env python3
"""
Simple script to download MLP data from Modal and create plots.
"""

import subprocess
import sys
import os

def download_mlp_data():
    """Download MLP data from the known volume"""
    print("📥 Downloading MLP data from Modal...")
    
    try:
        result = subprocess.run([
            "modal", "volume", "get", "mlp-csv-logs", "mlp_data/"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ MLP data downloaded successfully!")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Download timed out")
        return False
    except FileNotFoundError:
        print("❌ Modal CLI not found. Install with: pip install modal")
        return False

def main():
    print("🚀 MLP Data Downloader")
    print("=" * 30)
    
    # Download data
    if download_mlp_data():
        print("\n✅ Data ready! Now run:")
        print("   python analyze_mlp_results.py")
        
        # Check what we downloaded
        if os.path.exists("mlp_data"):
            print(f"\n📁 Downloaded to mlp_data/:")
            for root, dirs, files in os.walk("mlp_data"):
                level = root.replace("mlp_data", "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files)-5} more files")
    else:
        print("\n❌ Download failed. Try manually:")
        print("   modal volume get mlp-csv-logs mlp_data/")

if __name__ == "__main__":
    main()

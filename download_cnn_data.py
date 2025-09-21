#!/usr/bin/env python3
"""
Simple script to download CNN data from Modal and create plots.
"""

import subprocess
import sys
import os

def download_cnn_data():
    """Download CNN data from the known volume"""
    print("📥 Downloading CNN data from Modal...")
    
    try:
        result = subprocess.run([
            "modal", "volume", "get", "cnn-csv-logs", "cnn_data/"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ CNN data downloaded successfully!")
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
    print("🚀 CNN Data Downloader")
    print("=" * 30)
    
    # Download data
    if download_cnn_data():
        print("\n✅ Data ready! Now run:")
        print("   python analyze_cnn_results.py")
        
        # Check what we downloaded
        if os.path.exists("cnn_data"):
            print(f"\n📁 Downloaded to cnn_data/:")
            for root, dirs, files in os.walk("cnn_data"):
                level = root.replace("cnn_data", "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files)-5} more files")
    else:
        print("\n❌ Download failed. Try manually:")
        print("   modal volume get cnn-csv-logs cnn_data/")
        print("   or: python download_all_cnn_data.py")

if __name__ == "__main__":
    main()

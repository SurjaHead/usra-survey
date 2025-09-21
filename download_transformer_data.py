#!/usr/bin/env python3
"""
Transformer Data Downloader
Downloads all Transformer (BERT) timing data from Modal volume to local directory
"""

import subprocess
import sys
import os

def download_transformer_data():
    """Download Transformer data from Modal volume"""
    print("🚀 Transformer Data Downloader")
    print("=" * 35)
    
    # Create local directory
    local_dir = "transformer_data"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"📁 Created directory: {local_dir}/")
    else:
        print(f"📁 Using existing directory: {local_dir}/")
    
    # Download command
    volume_name = "bert-csv-logs"
    cmd = ["modal", "volume", "get", volume_name, "/", local_dir]
    
    print(f"📥 Downloading Transformer data from Modal...")
    print(f"   Volume: {volume_name}")
    print(f"   Target: {local_dir}/")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Download successful!")
        
        # List downloaded directories
        if os.path.exists(local_dir):
            dirs = [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]
            if dirs:
                print(f"📦 Downloaded {len(dirs)} Transformer run directories:")
                for i, dir_name in enumerate(sorted(dirs), 1):
                    print(f"   {i:2d}. {dir_name}")
            else:
                print("⚠️  No directories found in downloaded data")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e.stderr}")
        print("❌ Download failed. Try manually:")
        print(f"   modal volume get {volume_name} / {local_dir}")
        return False
    
    except FileNotFoundError:
        print("❌ Modal CLI not found. Please install Modal:")
        print("   pip install modal")
        return False
    
    return True

if __name__ == "__main__":
    success = download_transformer_data()
    sys.exit(0 if success else 1)

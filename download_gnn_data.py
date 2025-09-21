#!/usr/bin/env python3
"""
GNN Data Downloader
Downloads all GNN timing data from Modal volume to local directory
"""

import subprocess
import sys
import os

def download_gnn_data():
    """Download GNN data from Modal volume"""
    print("🚀 GNN Data Downloader")
    print("=" * 30)
    
    # Create local directory
    local_dir = "gnn_data"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"📁 Created directory: {local_dir}/")
    else:
        print(f"📁 Using existing directory: {local_dir}/")
    
    # Download command
    volume_name = "gnn-csv-logs"
    cmd = ["modal", "volume", "get", volume_name, "/", local_dir]
    
    print(f"📥 Downloading GNN data from Modal...")
    print(f"   Volume: {volume_name}")
    print(f"   Target: {local_dir}/")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Download successful!")
        
        # List downloaded directories
        if os.path.exists(local_dir):
            dirs = [d for d in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, d))]
            if dirs:
                print(f"📦 Downloaded {len(dirs)} GNN run directories:")
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
    success = download_gnn_data()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Helper script to check Modal volumes and download MLP data.
"""

import subprocess
import sys

def list_volumes():
    """List all available Modal volumes"""
    print("📋 Listing Modal volumes...")
    try:
        result = subprocess.run(["modal", "volume", "list"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("Available volumes:")
            print(result.stdout)
            return result.stdout
        else:
            print(f"❌ Error: {result.stderr}")
            return None
    except FileNotFoundError:
        print("❌ Modal CLI not found. Install with: pip install modal")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def download_volume(volume_name, local_dir="mlp_data"):
    """Download a specific volume"""
    print(f"📥 Downloading volume '{volume_name}' to '{local_dir}/'...")
    
    try:
        result = subprocess.run([
            "modal", "volume", "get", volume_name, f"{local_dir}/"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ Successfully downloaded {volume_name}")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Download timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🔍 Modal Volume Checker")
    print("=" * 30)
    
    # List volumes first
    volumes_output = list_volumes()
    
    if volumes_output is None:
        return
    
    # Extract volume names from the table format
    # Look for lines that contain volume names (not table borders/headers)
    volume_names = []
    for line in volumes_output.split('\n'):
        line = line.strip()
        # Skip empty lines, borders, and headers
        if not line or '━' in line or '┃' in line or 'Name' in line or '┏' in line or '┗' in line or '┡' in line:
            continue
        # Extract volume name from table row: │ volume-name │ date │
        if '│' in line:
            parts = [part.strip() for part in line.split('│') if part.strip()]
            if len(parts) >= 2:
                volume_name = parts[0]
                if volume_name and not volume_name.isspace():
                    volume_names.append(volume_name)
    
    # Filter for MLP/CSV related volumes
    mlp_volumes = [vol for vol in volume_names if any(keyword in vol.lower() for keyword in ['mlp', 'csv', 'log'])]
    
    if mlp_volumes:
        print(f"\n🎯 Found {len(mlp_volumes)} MLP-related volumes:")
        for i, volume in enumerate(mlp_volumes, 1):
            print(f"   {i}. {volume}")
        
        # Try to download MLP volume first
        for volume_name in mlp_volumes:
            if 'mlp' in volume_name.lower():
                print(f"\n🔍 Attempting to download: {volume_name}")
                if download_volume(volume_name):
                    print("✅ Download successful! You can now run:")
                    print("   python analyze_mlp_results.py")
                    return
    
    # If no MLP volumes, show all available volumes
    if volume_names:
        print(f"\n📋 All available volumes:")
        for volume in volume_names:
            print(f"   - {volume}")
        print(f"\n💡 To download manually:")
        print(f"   modal volume get mlp-csv-logs mlp_data/  # For MLP data")
        print(f"   modal volume get cnn-csv-logs cnn_data/  # For CNN data")
    else:
        print("\n❌ No MLP-related volumes found.")
        print("💡 You may need to:")
        print("   1. Run your MLP training first")
        print("   2. Check if volumes have different names")
        print("   3. Manually specify the volume name")

if __name__ == "__main__":
    main()

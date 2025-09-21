#!/usr/bin/env python3
"""
Script to run the MLP training with CUDA activation timing on Modal H100

Usage:
    # Run all activation functions
    modal run MLP.py
    
    # Or run this script directly
    python run_mlp_modal.py
"""

import subprocess
import sys
from pathlib import Path

def run_modal_mlp():
    """Run the MLP training on Modal"""
    try:
        # Check if modal is installed
        result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Modal CLI not found. Please install it with: pip install modal")
            return False
        
        print(f"✅ Found Modal CLI: {result.stdout.strip()}")
        
        # Check if MLP.py exists
        mlp_path = Path("MLP.py")
        if not mlp_path.exists():
            print("❌ MLP.py not found in current directory")
            return False
        
        print("🚀 Starting MLP training on Modal H100...")
        print("This will train MLPs with 8 different activation functions:")
        print("  - ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, SiLU, Mish")
        print("Each training run includes explicit CUDA timing for each activation layer.")
        print("Results will be saved as CSV files for easy analysis.")
        print()
        
        # Run the modal app
        cmd = ["modal", "run", "MLP.py"]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print("✅ All MLP training runs completed successfully!")
            print("📊 Check the tensorboard logs for activation timing analysis.")
            return True
        else:
            print(f"❌ Modal run failed with exit code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running Modal: {str(e)}")
        return False

def main():
    """Main function"""
    print("MLP CUDA Activation Timing - Modal H100 Runner")
    print("=" * 50)
    
    success = run_modal_mlp()
    
    if success:
        print("\n🎉 Training completed! Next steps:")
        print("1. Download the CSV logs from Modal")
        print("2. Run: python analyze_csv_results.py csv_logs/")
        print("3. View the generated visualizations and analysis report")
        print("4. Check the CSV files for detailed activation timing data")
    else:
        print("\n💥 Training failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

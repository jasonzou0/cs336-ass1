#!/usr/bin/env python3
"""
System Compute Information Script

This script detects and displays information about all available compute devices:
CPU, NVIDIA GPU (CUDA), AMD GPU, Intel GPU, and Apple Silicon (MPS).
"""

import torch
import platform
import psutil
import subprocess
import sys
import os

def get_system_info():
    """Get basic system information"""
    print("💻 System Information")
    print("=" * 50)
    
    print(f"  Operating System: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python Version: {platform.python_version()}")
    print(f"  PyTorch Version: {torch.__version__}")
    print()


def get_cpu_info():
    """Get CPU information"""
    print("🖥️  CPU Information")
    print("-" * 30)
    
    # CPU details
    print(f"  Processor: {platform.processor()}")
    print(f"  Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical Cores: {psutil.cpu_count(logical=True)}")
    
    # CPU frequency
    try:
        freq = psutil.cpu_freq()
        if freq:
            print(f"  Base Frequency: {freq.current:.0f} MHz")
            if freq.max > 0:
                print(f"  Max Frequency: {freq.max:.0f} MHz")
    except:
        pass
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
    print()


def check_cuda_gpu():
    """Check for NVIDIA CUDA GPUs"""
    print("🟢 NVIDIA CUDA GPU Check")
    print("-" * 30)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"  ✅ CUDA Available: {num_gpus} GPU(s) detected")
        
        for i in range(num_gpus):
            device = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {device.name}")
            print(f"    Compute Capability: {device.major}.{device.minor}")
            print(f"    Memory: {device.total_memory / (1024**3):.1f} GB")
            print(f"    Multiprocessors: {device.multi_processor_count}")
            
            # Estimate CUDA cores
            cuda_cores = estimate_cuda_cores(device)
            if cuda_cores:
                print(f"    Estimated CUDA Cores: {cuda_cores:,}")
    else:
        print("  ❌ CUDA not available")
        
        # Check why CUDA might not be available
        print("  Possible reasons:")
        print("    - No NVIDIA GPU installed")
        print("    - NVIDIA drivers not installed")
        print("    - PyTorch CPU-only version installed")
        print("    - CUDA toolkit version mismatch")
    print()


def check_mps_apple():
    """Check for Apple Metal Performance Shaders"""
    print("🍎 Apple MPS (Metal) Check")
    print("-" * 30)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✅ Apple MPS Available")
        print("  Apple Silicon GPU acceleration supported")
    else:
        print("  ❌ Apple MPS not available")
        print("  (This is normal on non-Apple Silicon devices)")
    print()


def check_other_devices():
    """Check for other compute devices"""
    print("🔧 Other Compute Devices")
    print("-" * 30)
    
    # Check for AMD ROCm
    try:
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print("  ✅ AMD ROCm detected")
        else:
            print("  ❌ AMD ROCm not available")
    except:
        print("  ❌ AMD ROCm not available")
    
    # Check for Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex
        print("  ✅ Intel Extension for PyTorch available")
    except ImportError:
        print("  ❌ Intel Extension for PyTorch not available")
    
    print()


def estimate_cuda_cores(device):
    """Estimate CUDA cores based on compute capability and SM count"""
    cores_per_sm = {
        (2, 0): 32, (2, 1): 48,   # Fermi
        (3, 0): 192, (3, 5): 192, (3, 7): 192,  # Kepler
        (5, 0): 128, (5, 2): 128,  # Maxwell
        (6, 0): 64, (6, 1): 128,   # Pascal
        (7, 0): 64, (7, 5): 64,    # Volta/Turing
        (8, 0): 64, (8, 6): 128, (8, 9): 128,   # Ampere/Ada
        (9, 0): 128,  # Hopper
    }
    
    arch = (device.major, device.minor)
    if arch in cores_per_sm:
        return device.multi_processor_count * cores_per_sm[arch]
    elif device.major >= 8:
        return device.multi_processor_count * 128
    elif device.major >= 6:
        return device.multi_processor_count * 64
    return None


def get_nvidia_smi_info():
    """Try to get NVIDIA GPU info via nvidia-smi"""
    print("📊 NVIDIA-SMI Information")
    print("-" * 30)
    
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        if lines and lines[0]:
            print("  ✅ nvidia-smi detected GPUs:")
            for line in lines:
                if line.strip():
                    print(f"    {line.strip()}")
        else:
            print("  ❌ No GPUs found via nvidia-smi")
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ nvidia-smi not found or failed")
        print("  Install NVIDIA drivers to use nvidia-smi")
    
    print()


def performance_recommendations():
    """Provide performance recommendations"""
    print("💡 Training Performance Recommendations")
    print("-" * 40)
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        memory_gb = device.total_memory / (1024**3)
        
        print("  🚀 NVIDIA GPU Training:")
        if memory_gb >= 24:
            print("     - Batch size: 8-16")
            print("     - Model size: Large models supported")
        elif memory_gb >= 12:
            print("     - Batch size: 4-8") 
            print("     - Model size: Medium models")
        else:
            print("     - Batch size: 2-4")
            print("     - Use gradient checkpointing")
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  🍎 Apple Silicon Training:")
        print("     - Use MPS backend for acceleration")
        print("     - Batch size: 4-8 (depending on RAM)")
        print("     - Unified memory architecture advantage")
    
    else:
        print("  🖥️  CPU Training:")
        cores = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"     - Use all {cores} CPU cores")
        print(f"     - Available RAM: {memory_gb:.1f} GB")
        print("     - Batch size: 1-4 (CPU training is slower)")
        print("     - Consider using smaller models")
        print("     - Enable CPU optimizations:")
        print("       export OMP_NUM_THREADS={}".format(cores))
    
    print()


def test_compute_device():
    """Test the available compute device"""
    print("🧪 Compute Device Test")
    print("-" * 25)
    
    # Determine best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "Apple MPS"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    print(f"  Using device: {device_name}")
    
    try:
        import time
        
        # Create test tensors
        size = 1000 if device.type == 'cpu' else 2000
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Time matrix multiplication
        start_time = time.time()
        z = torch.matmul(x, y)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        
        print(f"  ✅ Matrix multiplication ({size}x{size}) successful")
        print(f"     Time: {(end_time - start_time)*1000:.1f} ms")
        
        # Memory info
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"     GPU Memory Used: {allocated:.1f} MB")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
    
    print()


def main():
    """Main function"""
    get_system_info()
    get_cpu_info()
    check_cuda_gpu()
    check_mps_apple()
    check_other_devices()
    get_nvidia_smi_info()
    performance_recommendations()
    test_compute_device()


if __name__ == "__main__":
    main()
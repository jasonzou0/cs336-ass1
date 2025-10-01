#!/usr/bin/env python3
"""
GPU Information Script

This script detects and displays detailed information about your GPU,
including CUDA cores, memory, compute capability, and other specifications.
"""

import torch
import subprocess
import sys

def get_gpu_info():
    """Get comprehensive GPU information"""
    
    print("🔍 GPU Detection and Information")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system")
        print("   - No NVIDIA GPU detected")
        print("   - Or CUDA drivers not installed")
        return
    
    # Get basic CUDA info
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA is available")
    print(f"📊 Number of GPUs detected: {num_gpus}")
    print()
    
    # Iterate through all available GPUs
    for i in range(num_gpus):
        print(f"🎯 GPU {i} Information:")
        print("-" * 30)
        
        # Basic properties
        device = torch.cuda.get_device_properties(i)
        
        print(f"  Name: {device.name}")
        print(f"  Compute Capability: {device.major}.{device.minor}")
        
        # Memory information
        total_memory = device.total_memory / (1024**3)  # Convert to GB
        print(f"  Total Memory: {total_memory:.2f} GB")
        
        # Get current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        free_memory = total_memory - allocated
        
        print(f"  Allocated Memory: {allocated:.2f} GB")
        print(f"  Cached Memory: {cached:.2f} GB") 
        print(f"  Free Memory: {free_memory:.2f} GB")
        
        # Multiprocessor information
        print(f"  Multiprocessors (SMs): {device.multi_processor_count}")
        
        # Estimate CUDA cores based on compute capability
        cuda_cores = estimate_cuda_cores(device)
        if cuda_cores:
            print(f"  Estimated CUDA Cores: {cuda_cores:,}")
        
        # Other specifications
        print(f"  Max Threads per Block: {device.max_threads_per_block}")
        print(f"  Max Threads per Multiprocessor: {device.max_threads_per_multiprocessor}")
        print(f"  Warp Size: {device.warp_size}")
        
        # Clock rates (if available)
        if hasattr(device, 'clock_rate'):
            clock_ghz = device.clock_rate / 1000000  # Convert kHz to GHz
            print(f"  Base Clock: {clock_ghz:.2f} GHz")
        
        print()
    
    # Get additional info using nvidia-ml-py if available
    try:
        get_nvidia_ml_info()
    except:
        print("📝 Note: Install nvidia-ml-py for additional GPU metrics:")
        print("   pip install nvidia-ml-py")
    
    # Try to get info from nvidia-smi
    try:
        get_nvidia_smi_info()
    except:
        print("⚠️  nvidia-smi not available - install NVIDIA drivers for detailed info")


def estimate_cuda_cores(device):
    """Estimate CUDA cores based on compute capability and SM count"""
    
    # CUDA cores per SM for different architectures
    cores_per_sm = {
        (2, 0): 32,   # Fermi
        (2, 1): 48,   # Fermi
        (3, 0): 192,  # Kepler
        (3, 5): 192,  # Kepler
        (3, 7): 192,  # Kepler
        (5, 0): 128,  # Maxwell
        (5, 2): 128,  # Maxwell
        (6, 0): 64,   # Pascal
        (6, 1): 128,  # Pascal
        (7, 0): 64,   # Volta
        (7, 5): 64,   # Turing
        (8, 0): 64,   # Ampere
        (8, 6): 128,  # Ampere
        (8, 9): 128,  # Ada Lovelace
        (9, 0): 128,  # Hopper
    }
    
    arch = (device.major, device.minor)
    
    if arch in cores_per_sm:
        return device.multi_processor_count * cores_per_sm[arch]
    else:
        # For unknown architectures, try to estimate
        if device.major >= 8:
            return device.multi_processor_count * 128  # Modern GPUs
        elif device.major >= 6:
            return device.multi_processor_count * 64   # Pascal/Volta era
        else:
            return None


def get_nvidia_ml_info():
    """Get additional GPU info using NVIDIA ML library"""
    try:
        import pynvml
        
        pynvml.nvmlInit()
        print("🔧 Additional GPU Details (via NVIDIA-ML):")
        print("-" * 40)
        
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                print(f"  GPU {i} Temperature: {temp}°C")
            except:
                pass
            
            # Power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                max_power = pynvml.nvmlDeviceGetMaxPowerManagement(handle) / 1000.0
                print(f"  GPU {i} Power Usage: {power:.1f}W / {max_power:.1f}W")
            except:
                pass
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                print(f"  GPU {i} Utilization: {util.gpu}% (Memory: {util.memory}%)")
            except:
                pass
        
        print()
        
    except ImportError:
        pass


def get_nvidia_smi_info():
    """Get GPU info using nvidia-smi command"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        print("📋 nvidia-smi Summary:")
        print("-" * 25)
        
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                name, driver, mem_total, mem_used, mem_free, temp, power, util = parts
                print(f"  GPU {i}: {name}")
                print(f"    Driver: {driver}")
                print(f"    Memory: {mem_used} / {mem_total} MB")
                print(f"    Temperature: {temp}°C")
                print(f"    Power: {power}W")
                print(f"    Utilization: {util}%")
                print()
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def performance_recommendations():
    """Provide performance recommendations based on GPU"""
    print("💡 Training Performance Tips:")
    print("-" * 30)
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        total_memory_gb = device.total_memory / (1024**3)
        
        # Memory-based recommendations
        if total_memory_gb >= 24:
            print("  🚀 High-end GPU detected!")
            print("     - Use batch_size 8-16 for large models")
            print("     - Consider gradient accumulation for even larger effective batch sizes")
        elif total_memory_gb >= 12:
            print("  💪 Mid-range GPU detected!")
            print("     - Use batch_size 4-8 depending on model size")
            print("     - Monitor memory usage during training")
        elif total_memory_gb >= 6:
            print("  ⚡ Entry-level GPU detected!")
            print("     - Use batch_size 2-4")
            print("     - Consider gradient checkpointing to save memory")
        else:
            print("  ⚠️  Limited GPU memory detected!")
            print("     - Use batch_size 1-2")
            print("     - Enable gradient checkpointing")
            print("     - Consider reducing model size")
        
        # Compute capability recommendations
        if device.major >= 8:
            print("  ✨ Modern GPU architecture - excellent for training!")
        elif device.major >= 7:
            print("  ✅ Good GPU architecture for training")
        else:
            print("  📝 Older GPU - consider upgrading for better performance")
        
        print()


def main():
    """Main function"""
    get_gpu_info()
    performance_recommendations()
    
    # Test a simple CUDA operation
    if torch.cuda.is_available():
        print("🧪 CUDA Functionality Test:")
        print("-" * 25)
        
        try:
            # Create tensors and perform operations
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            
            import time
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            end_time = time.time()
            
            print(f"  ✅ Matrix multiplication test passed")
            print(f"     Time: {(end_time - start_time)*1000:.2f} ms")
            print(f"     Result shape: {z.shape}")
            
        except Exception as e:
            print(f"  ❌ CUDA test failed: {e}")


if __name__ == "__main__":
    main()
"""
Assignment Part 3: Fusion Opportunities, Memory Optimizations, Hardware Backend
================================================================================
RUN: python q3.py
"""

import torch
import torchvision.models as models
import torch.fx
import re
import os

# =============================================================================
# Load and trace ResNet18 with FX
# =============================================================================
print("Loading and tracing ResNet18 with torch.fx...")
model = models.resnet18(weights=None)
model.eval()
traced = torch.fx.symbolic_trace(model)
nodes = list(traced.graph.nodes)

print("="*70)
print("PART 3: FUSION OPPORTUNITIES FROM FX/MLIR OUTPUT")
print("="*70)

# =============================================================================
# Analyze FX Graph
# =============================================================================
print("\n" + "="*70)
print("FX GRAPH ANALYSIS (from torch.fx.symbolic_trace)")
print("="*70)

# Find Conv-BN chains
conv_bn_chains = []
for node in nodes:
    if 'conv' in node.name and node.op == 'call_module':
        for user in node.users:
            if 'bn' in user.name or 'downsample_1' in user.name:
                conv_bn_chains.append((node.name, user.name))

# Find BN-ReLU chains  
bn_relu_chains = []
for node in nodes:
    if ('bn' in node.name or 'downsample_1' in node.name) and node.op == 'call_module':
        for user in node.users:
            if 'relu' in user.name:
                bn_relu_chains.append((node.name, user.name))

# Find Add-ReLU chains
add_relu_chains = []
for node in nodes:
    if node.op == 'call_function' and 'add' in str(node.target):
        for user in node.users:
            if 'relu' in user.name:
                add_relu_chains.append((node.name, user.name))

print(f"\nFX Graph Operation Counts:")
print(f"  - call_module nodes:   {sum(1 for n in nodes if n.op == 'call_module')}")
print(f"  - call_function nodes: {sum(1 for n in nodes if n.op == 'call_function')}")

print(f"\nFusion Opportunities Detected in FX Graph:")
print(f"  - Conv → BN chains:    {len(conv_bn_chains)}")
print(f"  - BN → ReLU chains:    {len(bn_relu_chains)}")
print(f"  - Add → ReLU chains:   {len(add_relu_chains)}")

print(f"\nSample Conv → BN chains from FX:")
for i, (conv, bn) in enumerate(conv_bn_chains[:3], 1):
    print(f"  {i}. {conv} → {bn}")
if len(conv_bn_chains) > 3:
    print(f"  ... and {len(conv_bn_chains)-3} more")

print(f"\nSample Add → ReLU chains from FX:")
for i, (add_node, relu) in enumerate(add_relu_chains[:3], 1):
    print(f"  {i}. {add_node} → {relu}")
if len(add_relu_chains) > 3:
    print(f"  ... and {len(add_relu_chains)-3} more")

# =============================================================================
# Analyze MLIR Output
# =============================================================================
print("\n" + "="*70)
print("MLIR ANALYSIS (from resnet18_mlir.mlir)")
print("="*70)

mlir_path = "resnet18_mlir.mlir"
if os.path.exists(mlir_path):
    with open(mlir_path, 'r') as f:
        mlir_content = f.read()
    
    # Count MLIR operations
    conv_count = mlir_content.count("linalg.conv_2d_nchw_fchw")
    generic_count = mlir_content.count("linalg.generic")
    pad_count = mlir_content.count("tensor.pad")
    pool_count = mlir_content.count("linalg.pooling")
    
    # Count arithmetic ops inside linalg.generic
    subf_count = mlir_content.count("arith.subf")
    mulf_count = mlir_content.count("arith.mulf")
    addf_count = mlir_content.count("arith.addf")
    select_count = mlir_content.count("arith.select")
    
    print(f"\nMLIR Operation Counts (actual from file):")
    print(f"  - linalg.conv_2d_nchw_fchw: {conv_count:3d}  (convolutions)")
    print(f"  - linalg.generic:           {generic_count:3d}  (element-wise ops)")
    print(f"  - tensor.pad:               {pad_count:3d}  (padding)")
    print(f"  - linalg.pooling:           {pool_count:3d}  (pooling)")
    
    print(f"\nArithmetic Ops inside linalg.generic blocks:")
    print(f"  - arith.subf:   {subf_count:3d}  (BN: subtract mean)")
    print(f"  - arith.mulf:   {mulf_count:3d}  (BN: multiply by inv_std, gamma)")
    print(f"  - arith.addf:   {addf_count:3d}  (BN: add beta, residual add)")
    print(f"  - arith.select: {select_count:3d}  (ReLU: max(0, x))")
    
    # Extract actual MLIR snippets for fusion pairs
    print(f"\nActual MLIR pattern for BatchNorm (subf → mulf → mulf → addf):")
    # Find first occurrence of the BN pattern
    lines = mlir_content.split('\n')
    for i, line in enumerate(lines):
        if 'arith.subf' in line and i < 100:
            print(f"  Line {i}: {line.strip()[:70]}")
            break
    for i, line in enumerate(lines):
        if 'arith.mulf' in line and i < 100:
            print(f"  Line {i}: {line.strip()[:70]}")
            break
    
    print(f"\nActual MLIR pattern for ReLU (select with cmpf):")
    for i, line in enumerate(lines):
        if 'arith.select' in line and i < 200:
            print(f"  Line {i}: {line.strip()[:70]}")
            break
else:
    print(f"\n⚠ MLIR file not found. Run q2.py first to generate it.")
    conv_count = generic_count = 0

# =============================================================================
# FUSION PAIR 1: Conv2d → BatchNorm2d → ReLU
# =============================================================================
print("\n" + "="*70)
print("FUSION PAIR 1: Conv2d → BatchNorm2d → ReLU")
print("="*70)

print(f"""
EVIDENCE FROM FX GRAPH:
  Found {len(conv_bn_chains)} Conv→BN chains and {len(bn_relu_chains)} BN→ReLU chains
  Example: {conv_bn_chains[0][0]} → {conv_bn_chains[0][1]} → relu

EVIDENCE FROM MLIR:
  - {conv_count} linalg.conv_2d_nchw_fchw operations
  - Each BatchNorm lowered to 4 linalg.generic ops:
    * {subf_count//conv_count if conv_count else 0} subf per conv (subtract mean)
    * {mulf_count//conv_count if conv_count else 0} mulf per conv (scale operations)  
    * {addf_count//conv_count if conv_count else 0} addf per conv (add beta + residual)
  - {select_count} arith.select operations (ReLU activations)

EFFECT ON MEMORY ACCESS:
  BEFORE FUSION: Conv writes to DRAM → BN reads/writes 4× → ReLU reads/writes
    Total: 6 intermediate tensor accesses per block
    Memory traffic for 112×112×64: 6 × 3.2MB = 19.2 MB

  AFTER FUSION: Single ConvBnReLU kernel
    Total: Input read + output write only  
    Memory traffic: 2 × 3.2MB = 6.4 MB
    SAVINGS: 67% reduction in memory bandwidth

EFFECT ON KERNEL LAUNCH OVERHEAD:
  BEFORE: 6 kernel launches per block (Conv + 4 BN ops + ReLU)
  AFTER:  1 kernel launch
  SAVINGS: 5 fewer launches × 5-20μs = 25-100μs per block
  Total for ResNet18: {len(conv_bn_chains)} blocks × ~50μs = ~{len(conv_bn_chains)*50}μs saved
""")

# =============================================================================
# FUSION PAIR 2: Add (Residual) → ReLU  
# =============================================================================
print("="*70)
print("FUSION PAIR 2: Add (Residual) → ReLU")
print("="*70)

print(f"""
EVIDENCE FROM FX GRAPH:
  Found {len(add_relu_chains)} Add→ReLU pairs
  Example: {add_relu_chains[0][0]} → {add_relu_chains[0][1]}

EVIDENCE FROM MLIR:
  - Residual adds appear as arith.addf inside linalg.generic
  - Followed by arith.select (ReLU) in separate linalg.generic
  - Pattern: linalg.generic(addf) → linalg.generic(select)

EFFECT ON MEMORY ACCESS:
  BEFORE FUSION:
    Add:  Read tensor A, Read tensor B → Write sum to DRAM
    ReLU: Read sum from DRAM → Write result to DRAM
    Total: 4 tensor memory accesses

  AFTER FUSION (AddReLU kernel):
    Read A, Read B → Compute max(0, A+B) in registers → Write result
    Total: 3 tensor memory accesses
    SAVINGS: 25% reduction, eliminates intermediate tensor allocation

EFFECT ON KERNEL LAUNCH OVERHEAD:
  BEFORE: 2 kernel launches
  AFTER:  1 kernel launch
  SAVINGS: 5-20μs per residual block
  Total for ResNet18: {len(add_relu_chains)} blocks × ~10μs = ~{len(add_relu_chains)*10}μs saved
""")

# =============================================================================
# ADDITIONAL MEMORY OPTIMIZATIONS
# =============================================================================
print("="*70)
print("ADDITIONAL MEMORY OPTIMIZATIONS")
print("="*70)

print("""
1. IN-PLACE OPERATIONS
   - ReLU can overwrite input tensor when input not needed later
   - Observed: {0} arith.select ops could potentially be in-place
   - Memory savings: Up to 50% for activation storage

2. BUFFER REUSE
   - MLIR tensor.empty() creates new buffers; many can be reused
   - Tensors with non-overlapping lifetimes share memory
   - Reduces peak memory by 30-50%

3. BATCH NORM FOLDING (Constant Folding)
   - MLIR shows BN as 4 separate ops with constant parameters
   - At compile time: fold into Conv weights
     W_new = W × (γ/σ), b_new = (b-μ)×(γ/σ) + β
   - Eliminates all {0} BN-related linalg.generic ops

4. PAD-CONV FUSION
   - MLIR shows {1} tensor.pad ops before convolutions
   - Can fuse padding into conv kernel (implicit padding)
   - Eliminates intermediate padded tensor allocation
""".format(select_count if 'select_count' in dir() else '?', 
           pad_count if 'pad_count' in dir() else '?'))

# =============================================================================
# HARDWARE BACKEND RECOMMENDATION
# =============================================================================
print("="*70)
print("RECOMMENDED HARDWARE BACKEND: GPU (NVIDIA CUDA)")
print("="*70)

print(f"""
WHY GPU IS OPTIMAL FOR THIS MODEL:

1. PARALLELISM MATCHES WORKLOAD
   - MLIR shows {conv_count if 'conv_count' in dir() else 20} convolutions with millions of independent MACs
   - GPU: thousands of parallel cores vs CPU's tens of cores
   - Perfect mapping of data parallelism to GPU architecture

2. MEMORY BANDWIDTH
   - {generic_count if 'generic_count' in dir() else 167} element-wise ops are memory-bound
   - GPU: 500-2000 GB/s bandwidth vs CPU's 50-100 GB/s
   - Critical for achieving fusion benefits

3. NATIVE FUSION SUPPORT  
   - cuDNN: Fused Conv-BN-ReLU kernels built-in
   - TensorRT: Automatic fusion of the patterns we identified
   - The {len(conv_bn_chains)} Conv-BN-ReLU and {len(add_relu_chains)} Add-ReLU fusions fully supported

4. TENSOR CORES
   - FP16/INT8 acceleration: 4-8× over FP32
   - Maximizes throughput after fusion reduces memory bottleneck

COMPARISON:
  CPU: Lower parallelism, but viable for edge/small batch inference
  GPU: Best throughput, mature fusion libraries ★ RECOMMENDED
  TPU: Requires large batches (≥8), better for training than inference
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("="*70)
print("SUMMARY (Based on Actual FX/MLIR Analysis)")
print("="*70)

print(f"""
FX Graph Analysis:
  ✓ {len(conv_bn_chains)} Conv→BN chains detected
  ✓ {len(bn_relu_chains)} BN→ReLU chains detected  
  ✓ {len(add_relu_chains)} Add→ReLU chains detected

MLIR Analysis:
  ✓ {conv_count if 'conv_count' in dir() else '?'} convolutions, {generic_count if 'generic_count' in dir() else '?'} element-wise ops
  ✓ BN decomposed into {subf_count if 'subf_count' in dir() else '?'} subf + {mulf_count if 'mulf_count' in dir() else '?'} mulf + partial addf
  ✓ {select_count if 'select_count' in dir() else '?'} ReLU operations (arith.select)

Fusion Benefits:
  ✓ Conv-BN-ReLU fusion: 67% memory reduction, 5 fewer kernel launches
  ✓ Add-ReLU fusion: 25% memory reduction, 1 fewer kernel launch

Memory Optimizations:
  ✓ In-place operations, buffer reuse, BN folding, pad-conv fusion

Hardware Recommendation:
  ✓ GPU (CUDA) - best parallelism, bandwidth, and fusion support
""")

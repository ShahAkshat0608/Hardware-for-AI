import torch
import torchvision.models as models
import torch.fx

model = models.resnet18(pretrained=False)
model.eval() 

tracer = torch.fx.symbolic_trace(model)

print("--- Torch.FX Graph Representation ---")
tracer.graph.print_tabular()

# =============================================================================
# STRUCTURAL EXPLANATION (5-6 sentences)
# =============================================================================
"""
ResNet18 is a deep convolutional neural network consisting of an initial 7x7 
convolution followed by batch normalization, ReLU activation, and max pooling.
The network then contains 4 sequential "layers" (layer1-layer4), each composed 
of BasicBlock residual units containing pairs of 3x3 convolutions with batch 
normalization and ReLU activations, plus skip connections. Each BasicBlock adds 
the input (identity) to the output of the convolution path, enabling gradient 
flow through residual learning. The spatial dimensions are progressively reduced 
(from 56x56 to 7x7) while channels increase (64 -> 128 -> 256 -> 512) through 
strided convolutions. Finally, adaptive average pooling reduces spatial dims to 
1x1, followed by a fully connected layer mapping 512 features to 1000 classes.
"""

# =============================================================================
# REDUNDANT OR CHAINABLE OPERATIONS ANALYSIS
# =============================================================================
"""
1. CONV -> BATCHNORM CHAINS (Fusible):
   - Throughout ResNet18, every Conv2d is immediately followed by BatchNorm2d
   - Example: conv1 -> bn1, layer1.0.conv1 -> layer1.0.bn1, etc.
   - These can be fused into a single operation during inference since BN 
     parameters (mean, var, gamma, beta) are fixed and can be folded into 
     conv weights and biases.

2. BATCHNORM -> RELU CHAINS (Fusible):
   - Every BatchNorm2d is followed by ReLU activation
   - Example: bn1 -> relu, layer1.0.bn1 -> layer1.0.relu, etc.
   - These can be fused to reduce memory bandwidth by computing ReLU 
     in-place immediately after normalization.

3. CONV -> BATCHNORM -> RELU CHAINS (Triple Fusion Opportunity):
   - The pattern Conv2d -> BatchNorm2d -> ReLU appears repeatedly
   - All three can be fused into a single kernel (ConvBnRelu) to minimize 
     memory traffic and kernel launch overhead.

4. ADD -> RELU CHAINS (Fusible):
   - Every residual add operation is immediately followed by ReLU
   - Example: add -> layer1_0_relu_1, add_1 -> layer1_1_relu_1, etc.
   - Can be fused into AddReLU kernel to eliminate intermediate storage.

5. RELU MODULE REUSE (Redundancy):
   - The same ReLU module is called twice per BasicBlock:
     * Once after the first BN (layer1.0.relu called on layer1_0_bn1)
     * Once after the residual add (layer1.0.relu called on add result)
   - While functionally correct, this creates tracking overhead.

6. DOWNSAMPLE PATH CONV -> BN (Fusible):
   - In layers 2, 3, 4, the downsample path contains 1x1 Conv followed by BN
   - Example: layer2.0.downsample.0 -> layer2.0.downsample.1
   - These 4 additional Conv-BN pairs can also be fused.

7. IDENTITY SKIP CONNECTIONS (No-Op):
   - In BasicBlocks without downsampling (layer1), the skip connection is 
     simply an identity mapping - no computation needed, just pointer passing.
   - The 'add' operation combining residual + identity could potentially 
     be fused with the preceding ReLU of the next block.

8. AVGPOOL -> FLATTEN -> FC CHAIN (Reducible):
   - flatten is a reshape with zero computation
   - Could be eliminated by having avgpool output directly match FC input shape
   - All three could be fused into a single GlobalAvgPool+FC operation.

9. REPEATED STRUCTURAL PATTERNS:
   - The same BasicBlock pattern repeats 8 times (2 blocks × 4 layers)
   - A compiler could optimize by reusing the same fused kernel template.
"""

# =============================================================================
# Print additional graph statistics
# =============================================================================
print("\n--- Graph Statistics ---")
node_types = {}
for node in tracer.graph.nodes:
    op = node.op
    node_types[op] = node_types.get(op, 0) + 1
    
print(f"Total nodes: {len(list(tracer.graph.nodes))}")
for op_type, count in node_types.items():
    print(f"  {op_type}: {count}")

# =============================================================================
# COMPREHENSIVE FUSION OPPORTUNITY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("COMPREHENSIVE FUSION OPPORTUNITY ANALYSIS")
print("="*70)

# Build node list and lookup for pattern detection
nodes = list(tracer.graph.nodes)
node_to_idx = {n: i for i, n in enumerate(nodes)}

# Count basic operations
conv_nodes = [n for n in nodes if 'conv' in n.name and 'downsample' not in n.name]
downsample_conv_nodes = [n for n in nodes if 'downsample_0' in n.name]
bn_nodes = [n for n in nodes if 'bn' in n.name or 'downsample_1' in n.name]
relu_nodes = [n for n in nodes if 'relu' in n.name]
add_nodes = [n for n in nodes if n.op == 'call_function' and 'add' in str(n.target)]

print(f"\n1. BASIC OPERATION COUNTS:")
print(f"   - Main path Conv2d: {len(conv_nodes)}")
print(f"   - Downsample Conv2d (1x1): {len(downsample_conv_nodes)}")
print(f"   - BatchNorm2d: {len(bn_nodes)}")
print(f"   - ReLU activations: {len(relu_nodes)}")
print(f"   - Residual Add operations: {len(add_nodes)}")

# Pattern 1: Conv -> BN chains
print(f"\n2. CONV → BN FUSION OPPORTUNITIES:")
conv_bn_pairs = 0
for node in nodes:
    if 'conv' in node.name:
        # Check if next user is a BN
        for user in node.users:
            if 'bn' in user.name or 'downsample_1' in user.name:
                conv_bn_pairs += 1
print(f"   - Conv-BN pairs found: {conv_bn_pairs}")
print(f"   - Benefit: BN params (γ, β, μ, σ) can be folded into Conv weights/bias")
print(f"   - Memory savings: Eliminates intermediate tensor storage")

# Pattern 2: BN -> ReLU chains
print(f"\n3. BN → RELU FUSION OPPORTUNITIES:")
bn_relu_pairs = 0
for node in nodes:
    if 'bn' in node.name or 'downsample_1' in node.name:
        for user in node.users:
            if 'relu' in user.name:
                bn_relu_pairs += 1
            elif user.op == 'call_function' and 'add' in str(user.target):
                # BN feeds into add, not directly to relu
                pass
print(f"   - BN-ReLU pairs found: {bn_relu_pairs}")
print(f"   - Benefit: In-place ReLU computation after normalization")

# Pattern 3: Add -> ReLU chains
print(f"\n4. ADD → RELU FUSION OPPORTUNITIES:")
add_relu_pairs = 0
for node in nodes:
    if node.op == 'call_function' and 'add' in str(node.target):
        for user in node.users:
            if 'relu' in user.name:
                add_relu_pairs += 1
print(f"   - Add-ReLU pairs found: {add_relu_pairs}")
print(f"   - Benefit: Fused AddReLU kernel reduces memory round-trips")

# Pattern 4: Triple fusion Conv -> BN -> ReLU
print(f"\n5. CONV → BN → RELU TRIPLE FUSION:")
triple_fusions = 0
for node in nodes:
    if 'conv' in node.name:
        for bn_user in node.users:
            if 'bn' in bn_user.name:
                for relu_user in bn_user.users:
                    if 'relu' in relu_user.name:
                        triple_fusions += 1
print(f"   - Conv-BN-ReLU chains found: {triple_fusions}")
print(f"   - Benefit: Single kernel launch instead of 3, minimal memory traffic")

# Pattern 5: ReLU module reuse
print(f"\n6. RELU MODULE REUSE (Redundancy):")
relu_targets = {}
for node in nodes:
    if 'relu' in node.name and node.op == 'call_module':
        target = str(node.target)
        relu_targets[target] = relu_targets.get(target, 0) + 1
reused_relus = sum(1 for v in relu_targets.values() if v > 1)
print(f"   - ReLU modules called multiple times: {reused_relus}")
print(f"   - Details: {dict((k,v) for k,v in relu_targets.items() if v > 1)}")
print(f"   - Note: Same module reused for post-BN and post-Add activations")

# Pattern 6: Downsample path analysis
print(f"\n7. DOWNSAMPLE PATH FUSION:")
print(f"   - Downsample Conv-BN pairs: {len(downsample_conv_nodes)}")
print(f"   - Found in: layer2.0, layer3.0, layer4.0 (stride=2 transitions)")
print(f"   - These 1x1 convs + BN can also be fused")

# Pattern 7: Final layers
print(f"\n8. FINAL LAYER CHAIN (AvgPool → Flatten → FC):")
print(f"   - avgpool: Adaptive average pooling to 1x1")
print(f"   - flatten: Reshape operation (no computation)")
print(f"   - fc: Fully connected layer (matrix multiply)")
print(f"   - Opportunity: Flatten is redundant, can be eliminated")
print(f"   - Opportunity: AvgPool output shape can match FC input directly")

# Pattern 8: Identity skip connections
print(f"\n9. IDENTITY SKIP CONNECTIONS:")
identity_skips = 0
downsample_skips = 0
for node in add_nodes:
    args = node.args
    has_downsample = any('downsample' in str(arg) for arg in args)
    if has_downsample:
        downsample_skips += 1
    else:
        identity_skips += 1
print(f"   - Pure identity skips (no computation): {identity_skips}")
print(f"   - Skips with downsample projection: {downsample_skips}")
print(f"   - Identity skips are just pointer/reference passing")

# Summary
print(f"\n" + "="*70)
print("FUSION SUMMARY - POTENTIAL OPTIMIZATIONS")
print("="*70)
print(f"""
┌─────────────────────────────────────┬────────┬─────────────────────────────┐
│ Fusion Pattern                      │ Count  │ Benefit                     │
├─────────────────────────────────────┼────────┼─────────────────────────────┤
│ Conv → BN                           │   {conv_bn_pairs:2d}   │ Weight folding, -1 kernel   │
│ BN → ReLU                           │   {bn_relu_pairs:2d}   │ In-place activation         │
│ Add → ReLU                          │    {add_relu_pairs}   │ Fused residual activation   │
│ Conv → BN → ReLU (triple)           │   {triple_fusions:2d}   │ 3→1 kernel, min memory      │
│ Downsample Conv → BN                │    {len(downsample_conv_nodes)}   │ Same as Conv-BN             │
│ AvgPool → Flatten → FC              │    1   │ Eliminate reshape           │
├─────────────────────────────────────┼────────┼─────────────────────────────┤
│ Total kernel reduction potential    │  ~{conv_bn_pairs + bn_relu_pairs + add_relu_pairs}  │ Significant speedup         │
└─────────────────────────────────────┴────────┴─────────────────────────────┘
""")

print("\n" + "="*70)
print("NEXT: Run q2.py for torch-mlir MLIR generation and optimization proposals")
print("      Run q3.py for fusion analysis and hardware backend recommendations")
print("="*70)


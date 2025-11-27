"""
Debug training script with memory monitoring
Use this to identify where OOM occurs
"""

import sys
import torch
import gc

# Print GPU info
if torch.cuda.is_available():
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
    print()
else:
    print("WARNING: No GPU detected!")
    sys.exit(1)

def print_gpu_memory(stage):
    """Print current GPU memory usage"""
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"[{stage}] Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Test 1: Load data
print("=" * 60)
print("TEST 1: Loading data")
print("=" * 60)
try:
    from train import prepare_dataloaders

    print("Loading with conservative settings...")
    train_loader, val_loader, preprocessor = prepare_dataloaders(
        csv_path='../data/training_data.csv',
        batch_size=16,  # Very small batch
        val_split=0.2,
        max_rows=10000,  # Limit to 10K rows for debugging
        lookback=30,  # Smaller lookback
        num_workers=0  # No parallel loading for debugging
    )
    print(f"✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    sys.exit(1)

# Test 2: Create model
print("=" * 60)
print("TEST 2: Creating model")
print("=" * 60)
try:
    from models.transformer_lstm import create_model

    input_size = len(preprocessor.feature_columns)
    print(f"Input features: {input_size}")

    # Try lightweight first
    print("\nCreating lightweight_lstm model...")
    model = create_model(
        model_type='lightweight_lstm',
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    device = 'cuda'
    model = model.to(device)
    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print_gpu_memory("After model load")
    print()

except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Test 3: Single forward pass
print("=" * 60)
print("TEST 3: Single forward pass")
print("=" * 60)
try:
    # Get one batch
    X_batch, y_class, y_reg = next(iter(train_loader))
    print(f"Batch shape: {X_batch.shape}")

    X_batch = X_batch.to(device)
    y_class = y_class.to(device)
    y_reg = y_reg.to(device)

    print_gpu_memory("After data to GPU")

    # Forward pass
    with torch.no_grad():
        output = model(X_batch)
        print(f"✓ Forward pass successful")
        print(f"  Output shapes: {[o.shape for o in output]}")
        print_gpu_memory("After forward pass")

    del X_batch, y_class, y_reg, output
    torch.cuda.empty_cache()
    gc.collect()
    print()

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Training step with gradient
print("=" * 60)
print("TEST 4: Training step with gradients")
print("=" * 60)
try:
    import torch.nn as nn
    import torch.optim as optim

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # Get batch
    X_batch, y_class, y_reg = next(iter(train_loader))
    X_batch = X_batch.to(device)
    y_class = y_class.to(device)
    y_reg = y_reg.to(device).unsqueeze(1)

    # Forward + backward
    optimizer.zero_grad()
    class_logits, reg_pred = model(X_batch)

    loss_class = criterion_class(class_logits, y_class)
    loss_reg = criterion_reg(reg_pred, y_reg)
    loss = loss_class + 0.3 * loss_reg

    loss.backward()
    optimizer.step()

    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print_gpu_memory("After training step")

    del X_batch, y_class, y_reg, class_logits, reg_pred, loss
    torch.cuda.empty_cache()
    gc.collect()
    print()

except Exception as e:
    print(f"✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Full epoch
print("=" * 60)
print("TEST 5: Full training epoch")
print("=" * 60)
try:
    model.train()
    total_loss = 0

    for i, (X_batch, y_class, y_reg) in enumerate(train_loader):
        X_batch = X_batch.to(device, non_blocking=True)
        y_class = y_class.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        class_logits, reg_pred = model(X_batch)
        loss_class = criterion_class(class_logits, y_class)
        loss_reg = criterion_reg(reg_pred, y_reg)
        loss = loss_class + 0.3 * loss_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"  Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")
            print_gpu_memory(f"Batch {i}")

        # Clear memory periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    print(f"\n✓ Full epoch completed successfully")
    print(f"  Average loss: {avg_loss:.4f}")
    print_gpu_memory("After full epoch")
    print()

except Exception as e:
    print(f"✗ Full epoch failed at batch {i}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nYour GPU can handle training. If the main script still fails,")
print("try gradually increasing batch_size and model size.")
print("\nRecommended next steps:")
print("1. Start with lightweight_lstm model")
print("2. Use batch_size=16, then gradually increase to 32, 64")
print("3. Once stable, switch to transformer_lstm with small hidden_size=64")
print("4. Monitor 'nvidia-smi' during training to watch memory usage")

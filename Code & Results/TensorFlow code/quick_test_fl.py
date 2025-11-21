"""
Quick Test - Federated Learning (50 rounds instead of 200)
Run this for a faster test (~30-45 minutes)
"""

# Import the main FL code
import sys
sys.path.insert(0, '.')

# Modify parameters for quick testing
import sFLchain_vs_aFLchain_modern as fl_module

# Override parameters
fl_module.NUM_ROUNDS_FL = 50  # Quick test with 50 rounds instead of 200
fl_module.PARTITIONS = [100]  # Use 100 clients instead of 200 for faster testing
fl_module.PERCENTAGES = [0.75, 1]  # Still test both async and sync

print("="*60)
print("QUICK TEST MODE")
print("="*60)
print(f"Rounds: {fl_module.NUM_ROUNDS_FL} (instead of 200)")
print(f"Clients: {fl_module.PARTITIONS[0]} (instead of 200)")
print("Expected time: ~30-45 minutes")
print("="*60)
print()

# Run the main code
if __name__ == "__main__":
    # Configuration
    for num_clients in fl_module.PARTITIONS:
        print(f'\n{"#"*60}')
        print(f'# Dataset Partition: {num_clients} clients')
        print(f'{"#"*60}')
        
        for percentage in fl_module.PERCENTAGES:
            print(f'\n{"="*60}')
            print(f'Training with {percentage*100}% participation rate')
            print(f'{"="*60}')
            
            # Run federated learning
            results = fl_module.run_federated_learning(num_clients, percentage, fl_module.NUM_ROUNDS_FL)
            
            # Save results
            prefix = f"QUICKTEST_K{num_clients}_{percentage}"
            import numpy as np
            np.savetxt(f'train_loss_{prefix}.txt', 
                      np.reshape(results['train_loss'], (1, fl_module.NUM_ROUNDS_FL)))
            np.savetxt(f'train_accuracy_{prefix}.txt', 
                      np.reshape(results['train_accuracy'], (1, fl_module.NUM_ROUNDS_FL)))
            np.savetxt(f'test_loss_{prefix}.txt', 
                      np.reshape(results['test_loss'], (1, fl_module.NUM_ROUNDS_FL)))
            np.savetxt(f'test_accuracy_{prefix}.txt', 
                      np.reshape(results['test_accuracy'], (1, fl_module.NUM_ROUNDS_FL)))
            np.savetxt(f'eval_loss_{prefix}.txt', 
                      np.reshape(results['eval_loss'], (1, fl_module.NUM_ROUNDS_FL)))
            np.savetxt(f'eval_accuracy_{prefix}.txt', 
                      np.reshape(results['eval_accuracy'], (1, fl_module.NUM_ROUNDS_FL)))
            np.savetxt(f'iteration_time_{prefix}.txt', 
                      np.reshape(results['iteration_time'], (1, fl_module.NUM_ROUNDS_FL)))
            
            print(f"\n✓ Results saved with prefix: {prefix}\n")
    
    print(f'\n{"#"*60}')
    print("# Quick test completed successfully!")
    print(f'{"#"*60}\n')

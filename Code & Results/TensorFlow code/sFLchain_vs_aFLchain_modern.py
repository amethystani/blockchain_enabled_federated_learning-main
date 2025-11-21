"""
Modern Federated Learning Implementation (Python 3.12+ Compatible)
Replaces TensorFlow Federated with pure TensorFlow implementation
"""

import collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import random
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(1000)
tf.random.set_seed(1000)

# 0. PARAMETERS
NUM_CLIENTS_TEST = 50
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS_FL = 200
NUM_CLASSES_PER_USER = 10  # 10 = IID, <10 = non-IID

# Learning rates
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 1.00


# 1. FEDERATED LEARNING UTILITIES

class FederatedAveraging:
    """
    Simplified Federated Averaging implementation without TFF.
    Implements the FedAvg algorithm (McMahan et al., 2017).
    """
    
    def __init__(self, model_fn, num_clients):
        self.model_fn = model_fn
        self.num_clients = num_clients
        self.global_model = model_fn()
        self.client_optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT)
        
    def get_model_weights(self):
        """Get current global model weights"""
        return self.global_model.get_weights()
    
    def set_model_weights(self, weights):
        """Set global model weights"""
        self.global_model.set_weights(weights)
    
    def federated_averaging(self, client_weights, client_sizes):
        """
        Perform federated averaging of client model weights.
        
        Args:
            client_weights: List of model weights from each client
            client_sizes: List of dataset sizes for each client (for weighted averaging)
        
        Returns:
            Averaged model weights
        """
        total_size = sum(client_sizes)
        
        # Initialize averaged weights
        avg_weights = []
        
        # For each layer in the model
        for layer_idx in range(len(client_weights[0])):
            # Weighted average of this layer's weights across all clients
            layer_avg = np.zeros_like(client_weights[0][layer_idx])
            
            for client_idx, (weights, size) in enumerate(zip(client_weights, client_sizes)):
                layer_avg += weights[layer_idx] * (size / total_size)
            
            avg_weights.append(layer_avg)
        
        return avg_weights
    
    def client_update(self, model, dataset, dataset_size, epochs=NUM_EPOCHS):
        """
        Train a model on a client's local dataset.
        
        Args:
            model: Client's local model
            dataset: Client's local dataset
            dataset_size: Size of the dataset
            epochs: Number of local epochs
        
        Returns:
            Updated model weights, dataset size, and metrics
        """
        # CRITICAL: Create a NEW optimizer for THIS model (Keras 3.x requirement)
        # Cannot reuse optimizer across different models
        fresh_optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT)
        
        model.compile(
            optimizer=fresh_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Calculate steps per epoch based on dataset size and batch size
        # Since dataset repeats infinitely, we need to tell fit() when to stop
        steps_per_epoch = max(1, dataset_size // BATCH_SIZE)
        
        # Train on client's data with multiple epochs
        history = model.fit(
            dataset, 
            epochs=epochs, 
            steps_per_epoch=steps_per_epoch,
            verbose=0
        )
        
        # Get metrics from last epoch
        metrics = {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1]
        }
        
        return model.get_weights(), dataset_size, metrics
    
    def train_round(self, participating_datasets):
        """
        Execute one round of federated training.
        
        Args:
            participating_datasets: List of datasets for participating clients
        
        Returns:
            Training metrics for this round
        """
        client_weights = []
        client_sizes = []
        client_losses = []
        client_accuracies = []
        
        # Get current global weights
        global_weights = self.get_model_weights()
        
        # Train on each participating client
        for i, dataset in enumerate(participating_datasets):
            # Get dataset size (it's passed as a tuple or we need to look it up)
            # In this implementation, participating_datasets should be a list of (dataset, size) tuples
            ds, size = dataset
            
            # Create a fresh model for this client
            client_model = self.model_fn()
            client_model.set_weights(global_weights)
            
            # Update on client's data
            weights, _, metrics = self.client_update(client_model, ds, size)
            
            client_weights.append(weights)
            client_sizes.append(size)
            client_losses.append(metrics['loss'])
            client_accuracies.append(metrics['accuracy'])
        
        # Perform federated averaging
        new_weights = self.federated_averaging(client_weights, client_sizes)
        self.set_model_weights(new_weights)
        
        # Return aggregated metrics
        return {
            'train': {
                'loss': np.mean(client_losses),
                'sparse_categorical_accuracy': np.mean(client_accuracies)
            }
        }
    
    def evaluate(self, test_datasets):
        """
        Evaluate the global model on test datasets.
        
        Args:
            test_datasets: List of test datasets
        
        Returns:
            Evaluation metrics
        """
        # Create fresh optimizer for evaluation
        eval_optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT)
        
        self.global_model.compile(
            optimizer=eval_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        all_losses = []
        all_accuracies = []
        
        for dataset in test_datasets:
            # Evaluate on this dataset
            loss, accuracy = self.global_model.evaluate(dataset, verbose=0)
            all_losses.append(loss)
            all_accuracies.append(accuracy)
        
        return {
            'loss': np.mean(all_losses),
            'sparse_categorical_accuracy': np.mean(all_accuracies)
        }


# 2. DATA LOADING AND PREPROCESSING

def load_emnist_data():
    """Load EMNIST dataset using TensorFlow Datasets"""
    # Load EMNIST digits dataset
    train_ds, test_ds = tfds.load(
        'emnist/digits',
        split=['train', 'test'],
        as_supervised=False,
        shuffle_files=True
    )
    
    return train_ds, test_ds


def preprocess_dataset(dataset, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER):
    """
    Preprocess dataset for training.
    
    Args:
        dataset: Raw TensorFlow dataset
        batch_size: Batch size for training
        shuffle_buffer: Buffer size for shuffling
    
    Returns:
        Preprocessed dataset
    """
    def preprocess_fn(example):
        # Normalize pixel values to [0, 1]
        image = tf.cast(example['image'], tf.float32) / 255.0
        # Flatten image
        image = tf.reshape(image, [-1])
        label = example['label']
        return image, label
    
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.repeat()  # Repeat indefinitely to avoid running out during epochs
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def split_data_federated(dataset, num_clients, classes_per_client=10):
    """
    Split dataset into federated clients with controlled heterogeneity.
    
    Args:
        dataset: Full training dataset
        num_clients: Number of federated clients
        classes_per_client: Number of classes per client (10 = IID, <10 = non-IID)
    
    Returns:
        List of client datasets
    """
    # Convert to list for easier manipulation
    all_data = list(dataset.as_numpy_iterator())
    
    # Group by labels
    label_to_data = collections.defaultdict(list)
    for example in all_data:
        label = int(example['label'])
        label_to_data[label].append(example)
    
    # Create client datasets
    client_datasets = []
    client_sizes = []
    
    for client_id in range(num_clients):
        # Select random classes for this client
        if classes_per_client == 10:
            # IID: all classes
            selected_classes = list(range(10))
        else:
            # Non-IID: random subset of classes
            selected_classes = np.random.choice(10, classes_per_client, replace=False)
        
        # Collect data for this client
        client_data = []
        for class_id in selected_classes:
            class_data = label_to_data[class_id]
            # Take a subset of this class's data
            num_samples = len(class_data) // num_clients
            start_idx = (client_id * num_samples) % len(class_data)
            end_idx = start_idx + num_samples
            client_data.extend(class_data[start_idx:end_idx])
        
        # Shuffle client's data
        random.shuffle(client_data)
        
        # Convert to TensorFlow dataset
        def generator():
            for example in client_data:
                yield example
        
        output_signature = {
            'image': tf.TensorSpec(shape=(28, 28, 1), dtype=tf.uint8),
            'label': tf.TensorSpec(shape=(), dtype=tf.int64)
        }
        
        client_ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        client_datasets.append(client_ds)
        client_sizes.append(len(client_data))
    
    return client_datasets, client_sizes


# 3. MODEL DEFINITION

def create_keras_model():
    """Create the neural network model for EMNIST"""
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


# 4. VISUALIZATION

def visualize_client_data_distribution(client_datasets, num_clients_to_plot=6):
    """
    Visualize the data distribution across clients.
    
    Args:
        client_datasets: List of client datasets
        num_clients_to_plot: Number of clients to visualize
    """
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle("Label Counts for a Sample of Clients")
    
    for i in range(min(num_clients_to_plot, len(client_datasets))):
        plot_data = collections.defaultdict(list)
        
        # Count labels for this client
        for example in client_datasets[i].as_numpy_iterator():
            label = int(example['label'])
            plot_data[label].append(label)
        
        plt.subplot(2, 3, i + 1)
        plt.title(f"Client {i}")
        
        # Plot histogram for each class
        for class_id in range(10):
            if len(plot_data[class_id]) > 0:
                plt.hist(plot_data[class_id], bins=np.arange(0, 11), alpha=0.7)
        
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig("./test_figure.jpg")
    print("✓ Saved data distribution visualization to test_figure.jpg")


# 5. MAIN TRAINING LOOP

def run_federated_learning(num_clients, participation_rate, num_rounds=NUM_ROUNDS_FL):
    """
    Run the federated learning experiment.
    
    Args:
        num_clients: Total number of clients
        participation_rate: Fraction of clients participating in each round
        num_rounds: Number of communication rounds
    
    Returns:
        Dictionary containing all training metrics
    """
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Experiment")
    print(f"Number of Clients: {num_clients}")
    print(f"Participation Rate: {participation_rate}")
    print(f"Number of Rounds: {num_rounds}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading EMNIST dataset...")
    train_ds, test_ds = load_emnist_data()
    
    # Split into federated clients
    print(f"Creating {num_clients} federated clients...")
    client_datasets_raw, client_sizes = split_data_federated(train_ds, num_clients, NUM_CLASSES_PER_USER)
    
    # Preprocess all client datasets
    client_datasets = [preprocess_dataset(ds) for ds in client_datasets_raw]
    
    # Visualize data distribution
    print("Visualizing data distribution...")
    visualize_client_data_distribution(client_datasets_raw)
    
    # Prepare test datasets (one per client for evaluation)
    print("Preparing test datasets...")
    test_datasets_raw, _ = split_data_federated(test_ds, num_clients, 10)  # IID test data
    test_datasets = [preprocess_dataset(ds) for ds in test_datasets_raw]
    
    # Initialize federated learning
    print("Initializing federated learning system...")
    fed_avg = FederatedAveraging(create_keras_model, num_clients)
    
    # Metrics storage
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    eval_loss = []
    eval_accuracy = []
    iteration_time = []
    
    # Training loop
    print("\nStarting training...\n")
    num_participating = int(num_clients * participation_rate)
    
    for round_num in tqdm(range(num_rounds), desc="FL Rounds"):
        round_start = time.time()
        
        # Select random subset of clients
        participating_indices = np.random.choice(
            num_clients, 
            size=num_participating, 
            replace=False
        )
        
        participating_datasets = [(client_datasets[i], client_sizes[i]) for i in participating_indices]
        
        # Training round
        train_metrics = fed_avg.train_round(participating_datasets)
        
        # Store training metrics
        train_loss.append(train_metrics['train']['loss'])
        train_accuracy.append(train_metrics['train']['sparse_categorical_accuracy'])
        
        # Evaluate on test datasets of participating clients
        test_datasets_participating = [test_datasets[i] for i in participating_indices]
        test_metrics = fed_avg.evaluate(test_datasets_participating)
        test_loss.append(test_metrics['loss'])
        test_accuracy.append(test_metrics['sparse_categorical_accuracy'])
        
        # Evaluate on random sample of all clients
        eval_indices = np.random.choice(num_clients, size=NUM_CLIENTS_TEST, replace=False)
        eval_datasets = [test_datasets[i] for i in eval_indices]
        eval_metrics = fed_avg.evaluate(eval_datasets)
        eval_loss.append(eval_metrics['loss'])
        eval_accuracy.append(eval_metrics['sparse_categorical_accuracy'])
        
        # Track time
        round_end = time.time()
        iteration_time.append(round_end - round_start)
        
        # Print progress every 10 rounds
        if (round_num + 1) % 10 == 0:
            print(f"\nRound {round_num + 1}/{num_rounds}:")
            print(f"  Train Loss: {train_loss[-1]:.4f}, Train Acc: {train_accuracy[-1]:.4f}")
            print(f"  Test Loss: {test_loss[-1]:.4f}, Test Acc: {test_accuracy[-1]:.4f}")
            print(f"  Eval Loss: {eval_loss[-1]:.4f}, Eval Acc: {eval_accuracy[-1]:.4f}")
            print(f"  Round Time: {iteration_time[-1]:.2f}s")
    
    print("\n✓ Training complete!\n")
    
    # Return all metrics
    return {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,
        'iteration_time': iteration_time
    }


# 6. MAIN EXECUTION

if __name__ == "__main__":
    # Configuration
    PARTITIONS = [200]  # Number of clients
    PERCENTAGES = [0.75, 1]  # Participation rates (async, sync)
    
    # Run experiments
    for num_clients in PARTITIONS:
        print(f'\n{"#"*60}')
        print(f'# Dataset Partition: {num_clients} clients')
        print(f'{"#"*60}')
        
        for percentage in PERCENTAGES:
            print(f'\n{"="*60}')
            print(f'Training with {percentage*100}% participation rate')
            print(f'{"="*60}')
            
            # Run federated learning
            results = run_federated_learning(num_clients, percentage, NUM_ROUNDS_FL)
            
            # Save results
            prefix = f"K{num_clients}_{percentage}"
            np.savetxt(f'train_loss_{prefix}.txt', 
                      np.reshape(results['train_loss'], (1, NUM_ROUNDS_FL)))
            np.savetxt(f'train_accuracy_{prefix}.txt', 
                      np.reshape(results['train_accuracy'], (1, NUM_ROUNDS_FL)))
            np.savetxt(f'test_loss_{prefix}.txt', 
                      np.reshape(results['test_loss'], (1, NUM_ROUNDS_FL)))
            np.savetxt(f'test_accuracy_{prefix}.txt', 
                      np.reshape(results['test_accuracy'], (1, NUM_ROUNDS_FL)))
            np.savetxt(f'eval_loss_{prefix}.txt', 
                      np.reshape(results['eval_loss'], (1, NUM_ROUNDS_FL)))
            np.savetxt(f'eval_accuracy_{prefix}.txt', 
                      np.reshape(results['eval_accuracy'], (1, NUM_ROUNDS_FL)))
            np.savetxt(f'iteration_time_{prefix}.txt', 
                      np.reshape(results['iteration_time'], (1, NUM_ROUNDS_FL)))
            
            print(f"\n✓ Results saved with prefix: {prefix}\n")
    
    print(f'\n{"#"*60}')
    print("# All experiments completed successfully!")
    print(f'{"#"*60}\n')

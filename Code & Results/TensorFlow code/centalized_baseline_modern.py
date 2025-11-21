"""
Centralized Baseline for EMNIST (Python 3.12+ Compatible)
Simple centralized training without federated learning
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time

# Set random seeds
np.random.seed(1000)
tf.random.set_seed(1000)

# Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
SHUFFLE_BUFFER = 10000

print("="*60)
print("Centralized EMNIST Training Baseline")
print("="*60)

# Load EMNIST dataset
print("\nLoading EMNIST dataset...")
train_ds, test_ds = tfds.load(
    'emnist/digits',
    split=['train', 'test'],
    as_supervised=False,
    shuffle_files=True
)

# Preprocess function
def preprocess_fn(example):
    image = tf.cast(example['image'], tf.float32) / 255.0
    image = tf.reshape(image, [-1])  # Flatten to 784
    label = example['label']
    return image, label

# Prepare datasets
print("Preprocessing datasets...")
train_dataset = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Create model
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, kernel_initializer='zeros'),
    tf.keras.layers.Softmax(),
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Train model
print("\nTraining model...")
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    validation_data=test_dataset,
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

# Evaluate model
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Training Time: {training_time:.2f} seconds")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("="*60)

# Save results
print("\nSaving results...")
np.savetxt('centralized_train_loss.txt', np.array(history.history['loss']))
np.savetxt('centralized_train_accuracy.txt', np.array(history.history['accuracy']))
np.savetxt('centralized_val_loss.txt', np.array(history.history['val_loss']))
np.savetxt('centralized_val_accuracy.txt', np.array(history.history['val_accuracy']))

# Save model
model.save('centralized_baseline_model.h5')

print("✓ Training complete! Results saved.")

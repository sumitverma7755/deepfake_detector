import tensorflow as tf
import numpy as np
import time
from gpu_config import configure_gpu, get_device_strategy

def create_model(strategy):
    """
    Create a model within the strategy scope.
    """
    with strategy.scope():
        model = tf.keras.Sequential([
            # Customize your model architecture here
            tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def train_model(model, train_dataset, validation_dataset, epochs=10):
    """
    Train the model with GPU acceleration.
    """
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model_checkpoint.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            update_freq='batch'
        )
    ]
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1
    )
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    return history

def prepare_data(batch_size=32):
    """
    Prepare and optimize the datasets for GPU training.
    """
    # Example with MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Optimize for performance
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset

def main():
    # Configure GPU
    configure_gpu()
    
    # Get distribution strategy
    strategy = get_device_strategy()
    
    # Prepare data
    train_dataset, validation_dataset = prepare_data(batch_size=64)
    
    # Create model
    model = create_model(strategy)
    model.summary()
    
    # Train model
    history = train_model(model, train_dataset, validation_dataset, epochs=5)
    
    # Evaluate model
    loss, accuracy = model.evaluate(validation_dataset)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('gpu_trained_model.h5')
    print("Model saved successfully")

if __name__ == "__main__":
    main() 
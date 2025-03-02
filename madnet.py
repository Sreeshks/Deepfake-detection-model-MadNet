import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, LSTM, Dropout, TimeDistributed, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class DatasetPreparation:
    def __init__(self, input_shape=(299, 299, 3), seq_length=10):
        self.input_shape = input_shape
        self.seq_length = seq_length
        self.class_mapping = {
            'real': 0,
            'deepfake_face': 1,
            'deepfake_expression': 2,
            'deepfake_lighting': 3,
            'deepfake_blending': 4
        }

    def preprocess_image(self, img_path):
        """Load, resize, and normalize an image."""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.input_shape[:2])
        img = tf.keras.applications.xception.preprocess_input(img)
        return img

    def create_dataset(self, directory, batch_size=16):
        """Create a TensorFlow dataset with multinomial labels."""
        paths = []
        labels = []

        for class_name in self.class_mapping.keys():
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith('.png'):
                    paths.append(os.path.join(class_dir, img_name))
                    labels.append(self.class_mapping[class_name])

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = dataset.map(
            lambda path, label: (self.preprocess_image(path), tf.one_hot(label, depth=len(self.class_mapping))),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

class EnhancedMADNet:
    def __init__(self, input_shape=(299, 299, 3), seq_length=10, num_classes=5):
        self.input_shape = input_shape
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build the enhanced MADNet architecture with multinomial classification."""
        # Frame-level input
        frame_input = Input(shape=self.input_shape, name="Frame_Input")
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=self.input_shape)
        
        # Multi-scale feature extraction
        spatial_features = GlobalAveragePooling2D()(base_model(frame_input))
        spatial_features_2 = GlobalAveragePooling2D()(base_model(frame_input, training=False))

        # Sequence-level input
        sequence_input = Input(shape=(self.seq_length, *self.input_shape), name="Sequence_Input")
        time_distributed = TimeDistributed(base_model)(sequence_input)
        temporal_features = LSTM(256, return_sequences=False)(time_distributed)
        temporal_features_2 = LSTM(128, return_sequences=False)(time_distributed)

        # Combine all features
        combined = Concatenate()([spatial_features, spatial_features_2, temporal_features, temporal_features_2])
        x = BatchNormalization()(combined)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        output = Dense(self.num_classes, activation="softmax", name="Output")(x)

        return Model(inputs=[frame_input, sequence_input], outputs=output)

class Trainer:
    def __init__(self, model, learning_rate=0.0001):
        self.model = model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True)]
        )

    def train(self, train_dataset, valid_dataset, epochs=20, batch_size=16):
        """Train the model with provided datasets."""
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
        ]

        history = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history

    def evaluate(self, test_dataset):
        """Evaluate the model on the test dataset."""
        results = self.model.evaluate(test_dataset)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}, Test AUC: {results[2]:.4f}")
        return results

    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)
        print(f"Model saved to {path}")

# Example usage
if __name__ == "__main__":
    # Initialize components
    data_prep = DatasetPreparation()
    model = EnhancedMADNet()
    trainer = Trainer(model.model)

    # Create datasets (assuming data directory structure exists)
    train_dataset = data_prep.create_dataset("path/to/train/data")
    valid_dataset = data_prep.create_dataset("path/to/valid/data")
    test_dataset = data_prep.create_dataset("path/to/test/data")

    # Train the model
    trainer.train(train_dataset, valid_dataset)
    
    # Evaluate and save
    trainer.evaluate(test_dataset)
    trainer.save_model("enhanced_madnet_model.h5")
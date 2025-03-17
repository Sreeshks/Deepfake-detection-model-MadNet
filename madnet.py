import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.layers import (
    Input, Dense, GlobalAveragePooling2D, LSTM, Dropout, 
    TimeDistributed, Concatenate, BatchNormalization, 
    Conv2D, MaxPooling2D, Bidirectional, Attention, GRU,
    Conv3D, MaxPool3D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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
        
    def data_augmentation(self):
        """Create data augmentation pipeline for training"""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])

    def preprocess_image(self, img_path, augment=False):
        """Load, resize, and normalize an image with optional augmentation."""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.input_shape[:2])
        
        if augment:
            img = self.data_augmentation()(img)
            
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def create_dataset(self, directory, batch_size=16, augment=False):
        """Create a TensorFlow dataset with multinomial labels."""
        paths = []
        labels = []
        sequence_paths = []

        for class_name in self.class_mapping.keys():
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                continue

            # Get all image files in the class directory
            img_paths = []
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(class_dir, img_name))
            
            # Sort to ensure frames are in order for sequence creation
            img_paths.sort()
            
            # Create sequences of frames
            for i in range(len(img_paths) - self.seq_length + 1):
                sequence = img_paths[i:i+self.seq_length]
                sequence_paths.append(sequence)
                labels.append(self.class_mapping[class_name])
                # Add the middle frame as the reference frame
                paths.append(img_paths[i + self.seq_length // 2])

        # Create dataset for single frames
        frame_dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        frame_dataset = frame_dataset.map(
            lambda path, label: (self.preprocess_image(path, augment), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Create dataset for sequences
        def load_sequence(paths, label):
            sequence = tf.map_fn(
                lambda p: self.preprocess_image(p, augment),
                paths,
                fn_output_signature=tf.float32
            )
            return sequence, label
        
        sequence_dataset = tf.data.Dataset.from_tensor_slices((sequence_paths, labels))
        sequence_dataset = sequence_dataset.map(load_sequence, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Combine the datasets
        dataset = tf.data.Dataset.zip((frame_dataset, sequence_dataset))
        dataset = dataset.map(
            lambda frame_data, seq_data: ((frame_data[0], seq_data[0]), 
                                           tf.one_hot(frame_data[1], depth=len(self.class_mapping))),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if augment:
            # Use a larger buffer size for training
            buffer_size = min(10000, len(paths))
        else:
            buffer_size = min(1000, len(paths))
            
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
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
        
        # Multiple CNN backbones for ensemble effect
        efficientnet = EfficientNetB0(weights="imagenet", include_top=False, input_shape=self.input_shape)
        resnet = ResNet50V2(weights="imagenet", include_top=False, input_shape=self.input_shape)
        
        # Freeze early layers of the pre-trained models
        for layer in efficientnet.layers[:100]:
            layer.trainable = False
        for layer in resnet.layers[:100]:
            layer.trainable = False
        
        # Multi-scale spatial feature extraction
        efficient_features = efficientnet(frame_input)
        resnet_features = resnet(frame_input)
        
        # Additional convolutional layers for feature refinement
        conv_eff = Conv2D(256, (3, 3), activation='relu', padding='same')(efficient_features)
        conv_res = Conv2D(256, (3, 3), activation='relu', padding='same')(resnet_features)
        
        # Global pooling for spatial features
        spatial_features_eff = GlobalAveragePooling2D()(conv_eff)
        spatial_features_res = GlobalAveragePooling2D()(conv_res)
        
        # Sequence-level input
        sequence_input = Input(shape=(self.seq_length, *self.input_shape), name="Sequence_Input")
        
        # Temporal feature extraction
        # Option 1: CNN + LSTM approach
        time_distributed_eff = TimeDistributed(efficientnet)(sequence_input)
        time_distributed_pool = TimeDistributed(GlobalAveragePooling2D())(time_distributed_eff)
        
        # Bidirectional LSTM for temporal features
        temporal_features_bi = Bidirectional(LSTM(256, return_sequences=True))(time_distributed_pool)
        temporal_features_1 = Bidirectional(LSTM(128, return_sequences=False))(temporal_features_bi)
        
        # GRU for alternative temporal features
        temporal_features_gru = GRU(256, return_sequences=False)(time_distributed_pool)
        
        # Option 2: 3D CNN approach for spatio-temporal features
        # Reshape sequence input for 3D CNN
        reshaped_input = tf.keras.layers.Reshape((self.seq_length, self.input_shape[0], self.input_shape[1], self.input_shape[2]))(sequence_input)
        
        # 3D CNN layers
        conv3d_1 = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(reshaped_input)
        pool3d_1 = MaxPool3D(pool_size=(1, 2, 2))(conv3d_1)
        conv3d_2 = Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool3d_1)
        pool3d_2 = MaxPool3D(pool_size=(1, 2, 2))(conv3d_2)
        conv3d_3 = Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool3d_2)
        pool3d_3 = MaxPool3D(pool_size=(1, 2, 2))(conv3d_3)
        
        # Flatten 3D CNN output
        flattened_3d = tf.keras.layers.Reshape((-1, 256 * (self.input_shape[0] // 8) * (self.input_shape[1] // 8)))(pool3d_3)
        temporal_features_3d = LSTM(256, return_sequences=False)(flattened_3d)
        
        # Combine all features
        combined = Concatenate()([
            spatial_features_eff, 
            spatial_features_res, 
            temporal_features_1, 
            temporal_features_gru,
            temporal_features_3d
        ])
        
        # Dense layers with residual connections
        x = BatchNormalization()(combined)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        
        # First residual block
        res_1 = Dense(512, activation="relu")(x)
        res_1 = BatchNormalization()(res_1)
        res_1 = Dropout(0.5)(res_1)
        res_1 = Dense(512, activation="relu")(res_1)
        res_1 = BatchNormalization()(res_1)
        x = Concatenate()([x, res_1])
        
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)
        
        # Second residual block
        res_2 = Dense(256, activation="relu")(x)
        res_2 = BatchNormalization()(res_2)
        res_2 = Dropout(0.4)(res_2)
        res_2 = Dense(256, activation="relu")(res_2)
        res_2 = BatchNormalization()(res_2)
        x = Concatenate()([x, res_2])
        
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        output = Dense(self.num_classes, activation="softmax", name="Output")(x)

        return Model(inputs=[frame_input, sequence_input], outputs=output)

class Trainer:
    def __init__(self, model, learning_rate=0.0001):
        self.model = model
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy", 
                tf.keras.metrics.AUC(multi_label=True),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        self.history = None

    def train(self, train_dataset, valid_dataset, epochs=30, batch_size=16):
        """Train the model with provided datasets."""
        # Create a directory for model checkpoints
        checkpoint_dir = "model_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min"
            )
        ]

        self.history = self.model.fit(
            train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return self.history

    def evaluate(self, test_dataset):
        """Evaluate the model on the test dataset."""
        results = self.model.evaluate(test_dataset)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        print(f"Test AUC: {results[2]:.4f}")
        print(f"Test Precision: {results[3]:.4f}")
        print(f"Test Recall: {results[4]:.4f}")
        return results
    
    def predict_and_analyze(self, test_dataset):
        """Make predictions and analyze results with confusion matrix and classification report."""
        # Get true labels
        true_labels = []
        for _, labels in test_dataset:
            true_labels.extend(np.argmax(labels.numpy(), axis=1))
        
        # Make predictions
        predictions = self.model.predict(test_dataset)
        pred_labels = np.argmax(predictions, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        class_names = list(DatasetPreparation().class_mapping.keys())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Generate classification report
        report = classification_report(true_labels, pred_labels, target_names=class_names)
        print("Classification Report:")
        print(report)
        
        # Save report to file
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        
        return cm, report

    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
     # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        # Optionally plot AUC, Precision, and Recall
        plt.figure(figsize=(15, 5))
        plt.plot(self.history.history['auc'], label='Training AUC')
        plt.plot(self.history.history['val_auc'], label='Validation AUC')
        plt.plot(self.history.history['precision'], label='Training Precision')
        plt.plot(self.history.history['val_precision'], label='Validation Precision')
        plt.plot(self.history.history['recall'], label='Training Recall')
        plt.plot(self.history.history['val_recall'], label='Validation Recall')
        plt.title('Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig('model_metrics.png')
        plt.close()


class DeepfakeDetectionSystem:
    def __init__(self, model_path=None, input_shape=(299, 299, 3), seq_length=10):
        self.input_shape = input_shape
        self.seq_length = seq_length
        self.class_mapping = {
            0: 'real',
            1: 'deepfake_face',
            2: 'deepfake_expression',
            3: 'deepfake_lighting',
            4: 'deepfake_blending'
        }
        
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = EnhancedMADNet(input_shape, seq_length).model
    
    def preprocess_video(self, video_path, output_dir):
        """Extract frames from video and preprocess them."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use ffmpeg to extract frames (command-line approach)
        os.system(f"ffmpeg -i {video_path} -vf fps=30 {output_dir}/frame_%04d.png")
        
        # Collect and sort frame paths
        frame_paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')]
        frame_paths.sort()
        
        return frame_paths
    
    def create_sequences(self, frame_paths):
        """Create sequences from frame paths."""
        sequences = []
        frames = []
        
        for i in range(0, len(frame_paths) - self.seq_length + 1, self.seq_length // 2):
            sequence = frame_paths[i:i+self.seq_length]
            if len(sequence) == self.seq_length:
                sequences.append(sequence)
                frames.append(sequence[self.seq_length // 2])  # Middle frame
        
        return frames, sequences
    
    def preprocess_frames(self, frame_paths):
        """Preprocess frames."""
        preprocessed_frames = []
        for path in frame_paths:
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, self.input_shape[:2])
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            preprocessed_frames.append(img)
        
        return np.array(preprocessed_frames)
    
    def predict_video(self, video_path, temp_dir="temp_frames"):
        """Predict if a video is real or deepfake and the type of deepfake."""
        # Extract frames
        frame_paths = self.preprocess_video(video_path, temp_dir)
        
        # Create sequences
        single_frames, sequences = self.create_sequences(frame_paths)
        
        # Preprocess frames
        processed_frames = self.preprocess_frames(single_frames)
        
        # Preprocess sequences
        processed_sequences = []
        for seq in sequences:
            processed_sequences.append(self.preprocess_frames(seq))
        
        processed_sequences = np.array(processed_sequences)
        
        # Make predictions
        predictions = []
        for i in range(len(processed_frames)):
            frame = processed_frames[i:i+1]
            sequence = processed_sequences[i:i+1]
            pred = self.model.predict([frame, sequence])
            predictions.append(pred[0])
        
        # Aggregate predictions
        avg_prediction = np.mean(predictions, axis=0)
        class_probs = {self.class_mapping[i]: float(avg_prediction[i]) for i in range(len(avg_prediction))}
        
        final_class = self.class_mapping[np.argmax(avg_prediction)]
        max_prob = float(np.max(avg_prediction))
        
        result = {
            "prediction": final_class,
            "confidence": max_prob,
            "class_probabilities": class_probs,
            "is_deepfake": final_class != "real"
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # Set hyperparameters
    INPUT_SHAPE = (299, 299, 3)
    SEQ_LENGTH = 10
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    
    # Initialize components
    data_prep = DatasetPreparation(input_shape=INPUT_SHAPE, seq_length=SEQ_LENGTH)
    madnet = EnhancedMADNet(input_shape=INPUT_SHAPE, seq_length=SEQ_LENGTH)
    trainer = Trainer(madnet.model, learning_rate=LEARNING_RATE)

    # Create datasets with augmentation for training
    train_dataset = data_prep.create_dataset("path/to/train/data", batch_size=BATCH_SIZE, augment=True)
    valid_dataset = data_prep.create_dataset("path/to/valid/data", batch_size=BATCH_SIZE)
    test_dataset = data_prep.create_dataset("path/to/test/data", batch_size=BATCH_SIZE)

    # Train the model
    trainer.train(train_dataset, valid_dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate the model
    trainer.evaluate(test_dataset)
    
    # Detailed analysis
    trainer.predict_and_analyze(test_dataset)
    
    # Save the model
    trainer.save_model("enhanced_madnet_model.h5")
    
    # Create a detection system
    detection_system = DeepfakeDetectionSystem("enhanced_madnet_model.h5")
    
    # Test on a video
    result = detection_system.predict_video("path/to/test/video.mp4")
    print(result)

import os

class Config:
    """Configuration settings for the BERT model training."""
    
    # File paths
    DATA_PATH = os.path.join("data", "processed", "train.csv")
    OUTPUT_DIR = os.path.join("models", "saved_models")
    LOG_DIR = os.path.join("logs")
    
    # Training parameters
    EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    
    # Model parameters
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 6  # Number of labels for multi-label classification
    
    # Logging settings
    LOGGING_STEPS = 50
    SAVE_STEPS = 100
    EVAL_STEPS = 100
    
    # Other settings
    SEED = 42
    FP16 = True  # Use mixed precision training if available
    GRADIENT_ACCUMULATION_STEPS = 2
    WARMUP_RATIO = 0.1
    DATALOADER_NUM_WORKERS = 4
    DATALOADER_PIN_MEMORY = True
    
    @staticmethod
    def get_class_weights(labels):
        """Compute class weights for imbalanced datasets."""
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=labels)
        return class_weights.tolist()
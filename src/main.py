import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from src.trainer import Trainer
from src.data import ToxicCommentDataset
from src.config import Config
from src.utils import setup_logging, clean_text

def main():
    # Setup logging
    logger = setup_logging(log_dir="./logs")

    # Load and prepare data
    logger.info("Loading data...")
    df = pd.read_csv(os.path.join(Config.RAW_DATA_PATH, "train.csv"))
    df['cleaned_comment'] = df['comment_text'].apply(clean_text)

    # Split data
    logger.info("Splitting data into training and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['cleaned_comment'].values,
        df[Config.LABEL_COLUMNS].values,
        test_size=0.1,
        random_state=Config.SEED,
        stratify=df[Config.LABEL_COLUMNS].values
    )

    # Initialize tokenizer and model
    logger.info("Initializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.LABEL_COLUMNS),
        problem_type="multi_label_classification"
    )

    # Create datasets
    train_dataset = ToxicCommentDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicCommentDataset(val_texts, val_labels, tokenizer)

    # Initialize trainer
    trainer = Trainer(model=model, train_dataset=train_dataset, eval_dataset=val_dataset)

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate the model
    logger.info("Evaluating the model...")
    results = trainer.evaluate()
    logger.info(f"Final evaluation results: {results}")

if __name__ == "__main__":
    main()
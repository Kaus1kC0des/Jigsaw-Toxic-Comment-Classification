import os
import re
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json
import gc

# Suppress warnings and configure environment
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_logging(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logger = logging.getLogger('toxic_classifier')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def print_gpu_memory():
    if torch.cuda.is_available():
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).numpy()
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def save_model_artifacts(model, tokenizer, results, output_dir):
    final_output_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    
    logger.info(f"Saving model to {final_output_dir}")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    metadata = {
        'final_metrics': results,
        'model_config': model.config.to_dict()
    }
    
    metadata_path = os.path.join(final_output_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Model artifacts saved successfully")

def train_toxic_classifier(data_path, output_dir="./models/saved_models", epochs=3, batch_size=32):
    start_time = datetime.datetime.now()
    logger.info("Starting training...")
    os.makedirs(output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory()
    
    df = pd.read_csv(data_path)
    df['cleaned_comment'] = df['comment_text'].apply(clean_text)
    
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    labels = df[label_columns].values
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['cleaned_comment'].values,
        labels,
        test_size=0.1,
        random_state=SEED,
        stratify=labels[:, 0]
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=6,
        problem_type="multi_label_classification"
    )
    model = model.cuda() if torch.cuda.is_available() else model
    
    train_dataset = ToxicCommentDataset(train_texts, train_labels, tokenizer)
    val_dataset = ToxicCommentDataset(val_texts, val_labels, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        dataloader_drop_last=True,
        fp16=True,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    results = trainer.evaluate()
    logger.info(f"Final evaluation results: {results}")
    
    save_model_artifacts(model, tokenizer, results, output_dir)
    
    return model, tokenizer, results

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

if __name__ == "__main__":
    data_path = "../data/raw/train.csv"
    model, tokenizer, results = train_toxic_classifier(data_path=data_path)
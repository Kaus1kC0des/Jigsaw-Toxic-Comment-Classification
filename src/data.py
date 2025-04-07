import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class ToxicCommentDataset(Dataset):
    """Dataset class for toxic comment classification."""
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

def load_data(data_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(data_path)
    df['cleaned_comment'] = df['comment_text'].apply(clean_text)
    
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    labels = df[label_columns].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['cleaned_comment'].values,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels[:, 0]
    )

    return train_texts, train_labels, val_texts, val_labels

def clean_text(text):
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
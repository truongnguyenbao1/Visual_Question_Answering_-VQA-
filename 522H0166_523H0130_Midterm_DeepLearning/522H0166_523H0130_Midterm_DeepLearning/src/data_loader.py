import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, IMAGE_DIR, BATCH_SIZE, MAX_QUESTION_LEN, NUM_ANSWERS

# Download punkt tokenizer data quietly (using punkt, the old implementation used punkt_tab, but punkt is standard)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class VQADataset(Dataset):
    def __init__(self, data_samples, question_vocab=None, answer_vocab=None, max_len=MAX_QUESTION_LEN, num_answers=NUM_ANSWERS, image_dir=IMAGE_DIR):
        self.data = data_samples
        self.max_len = max_len
        self.image_dir = image_dir
        self.tokenizer = nltk.word_tokenize

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if question_vocab is None:
            self.build_question_vocab()
        else:
            self.question_vocab = question_vocab

        if answer_vocab is None:
            self.build_answer_vocab(num_answers)
        else:
            self.answer_vocab = answer_vocab

    def build_question_vocab(self):
        questions = [sample['question'].lower() for sample in self.data]
        all_text = ' '.join(questions)
        tokens = self.tokenizer(all_text)
        counter = Counter(tokens)
        vocab_size = min(10000, len(counter))
        self.question_vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.question_vocab.update({word: idx + 4 for idx, (word, _) in enumerate(counter.most_common(vocab_size - 4))})

    def build_answer_vocab(self, num_answers=1000):
        answers = [sample['answer'].lower() for sample in self.data]
        counter = Counter(answers)
        self.answer_vocab = {'<UNK>': 0}
        self.answer_vocab.update({ans: idx + 1 for idx, (ans, _) in enumerate(counter.most_common(num_answers - 1))})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_path = os.path.join(self.image_dir, sample['image_path'] + '.png')
        
        # Fallback to empty tensor if image not found (for local testing without data)
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception:
            # Placeholder in case image is missing
            image_tensor = torch.zeros((3, 224, 224))
            
        question = sample['question'].lower()
        tokens = ['<START>'] + self.tokenizer(question) + ['<END>']

        q_indices = [self.question_vocab.get(token, self.question_vocab['<UNK>']) for token in tokens]
        if len(q_indices) < self.max_len:
            q_indices += [self.question_vocab['<PAD>']] * (self.max_len - len(q_indices))
        else:
            q_indices = q_indices[:self.max_len]
        question_tensor = torch.tensor(q_indices, dtype=torch.long)

        answer_label = self.answer_vocab.get(sample['answer'].lower(), self.answer_vocab['<UNK>'])
        answer_tensor = torch.tensor(answer_label, dtype=torch.long)

        return {
            'image_path': image_tensor,
            'image_id': sample['image_path'],
            'question': question_tensor,
            'question_text': question,
            'answer': answer_tensor,
            'answer_text': sample['answer']
        }

def prepare_data(data_path=DATA_PATH):
    """
    Reads data.csv, oversamples rare classes, and splits into train and val datasets.
    """
    if not os.path.exists(data_path):
        print(f"File {data_path} not found. Please provide the dataset.")
        return [], []
        
    df = pd.read_csv(data_path)
    print(f"Number of main data samples: {len(df)}")

    X = df[['image_id', 'question']]
    y = df['answer']

    class_counts = Counter(y)
    rare_classes = {cls: count for cls, count in class_counts.items() if count < 2}
    print(f"Rare classes (<2): {len(rare_classes)}")

    df_oversampled = df.copy()
    for cls in rare_classes:
        rare_df = df[df['answer'] == cls]
        oversampled_rare = rare_df.sample(n=len(rare_df) * 2, replace=True, random_state=42)
        df_oversampled = pd.concat([df_oversampled, oversampled_rare], ignore_index=True)

    print(f"After oversample: {len(df_oversampled)} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        df_oversampled[['image_id', 'question']], df_oversampled['answer'],
        test_size=0.2, random_state=42, stratify=df_oversampled['answer']
    )

    train_data = []
    for idx in range(len(X_train)):
        row = X_train.iloc[idx]
        train_data.append({
            'image_path': row['image_id'],
            'question': row['question'] + '?' if not row['question'].endswith('?') else row['question'],
            'answer': y_train.iloc[idx]
        })

    val_data = []
    for idx in range(len(X_val)):
        row = X_val.iloc[idx]
        val_data.append({
            'image_path': row['image_id'],
            'question': row['question'] + '?' if not row['question'].endswith('?') else row['question'],
            'answer': y_val.iloc[idx]
        })

    return train_data, val_data

def get_dataloaders():
    train_data, val_data = prepare_data()
    
    if not train_data:
        return None, None, None, None

    train_dataset = VQADataset(train_data)
    val_dataset = VQADataset(val_data, question_vocab=train_dataset.question_vocab,
                             answer_vocab=train_dataset.answer_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset

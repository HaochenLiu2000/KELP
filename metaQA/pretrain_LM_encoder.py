import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
import json
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
parser = argparse.ArgumentParser(description="Parsing input arguments.")
parser.add_argument('--hop', type=int, required=True)
args = parser.parse_args()
hop=args.hop


model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
question_model = DistilBertModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_model.to(device)

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if len(sample['pos_triplet'])==3:
            if sample['pos_triplet'][1][0]=='~':
                pos=sample['pos_triplet'][2]+' '+sample['pos_triplet'][1][1:]+' '+sample['pos_triplet'][0]+'.'
            else:
                pos=sample['pos_triplet'][0]+' '+sample['pos_triplet'][1]+' '+sample['pos_triplet'][2]+'.'
        elif len(sample['pos_triplet'])==5:
            if sample['pos_triplet'][1][0]=='~':
                pos=sample['pos_triplet'][2]+' '+sample['pos_triplet'][1][1:]+' '+sample['pos_triplet'][0]+', '
            else:
                pos=sample['pos_triplet'][0]+' '+sample['pos_triplet'][1]+' '+sample['pos_triplet'][2]+', '
            if sample['pos_triplet'][3][0]=='~':
                pos+=sample['pos_triplet'][4]+' '+sample['pos_triplet'][3][1:]+' '+sample['pos_triplet'][2]+'.'
            else:
                pos+=sample['pos_triplet'][2]+' '+sample['pos_triplet'][3]+' '+sample['pos_triplet'][4]+'.'
            
        
        if len(sample['neg_triplet'])==3:
            if sample['neg_triplet'][1][0]=='~':
                neg=sample['neg_triplet'][2]+' '+sample['neg_triplet'][1][1:]+' '+sample['neg_triplet'][0]+'.'
            else:
                neg=sample['neg_triplet'][0]+' '+sample['neg_triplet'][1]+' '+sample['neg_triplet'][2]+'.'
        elif len(sample['neg_triplet'])==5:
            if sample['neg_triplet'][1][0]=='~':
                neg=sample['neg_triplet'][2]+' '+sample['neg_triplet'][1][1:]+' '+sample['neg_triplet'][0]+', '
            else:
                neg=sample['neg_triplet'][0]+' '+sample['neg_triplet'][1]+' '+sample['neg_triplet'][2]+', '
            if sample['neg_triplet'][3][0]=='~':
                neg+=sample['neg_triplet'][4]+' '+sample['neg_triplet'][3][1:]+' '+sample['neg_triplet'][2]+'.'
            else:
                neg+=sample['neg_triplet'][2]+' '+sample['neg_triplet'][3]+' '+sample['neg_triplet'][4]+'.'
        
        return {
            "question": sample["question"],
            "positive_question": pos,
            "negative_question": neg
        }
        
    def load_data(self, data_path):
        data = []
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                data.append(obj)
        return data


if hop==1:
    data_path = "onehop.jsonl"
    data_path_dev = "onehop-dev.jsonl"
if hop==2:
    data_path = "twohop.jsonl"
    data_path_dev = "twohop-dev.jsonl"

dataset = CustomDataset(data_path)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
optimizer_question = torch.optim.AdamW(question_model.parameters(), lr=2e-6)




dataset_dev = CustomDataset(data_path_dev)
dataloader_dev=DataLoader(dataset_dev, batch_size=5, shuffle=True)
criterion=nn.CosineSimilarity()
margin = 0.5

num_epochs = 60
for epoch in range(num_epochs):
    question_model.train()
    total_loss = 0
    total_loss_dev = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        
        question_input = tokenizer(batch["question"], return_tensors="pt", padding=True, truncation=True).to(device)
        positive_input = tokenizer(batch["positive_question"], return_tensors="pt", padding=True, truncation=True).to(device)
        negative_input = tokenizer(batch["negative_question"], return_tensors="pt", padding=True, truncation=True).to(device)
        
        question_embedding = question_model(**question_input).last_hidden_state.mean(dim=1)
        positive_embedding = question_model(**positive_input).last_hidden_state.mean(dim=1)
        negative_embedding = question_model(**negative_input).last_hidden_state.mean(dim=1)
        similarity_scores_pos=criterion(question_embedding, positive_embedding).mean()
        similarity_scores_neg=criterion(question_embedding, negative_embedding).mean()
        
        loss = torch.mean(torch.relu(1 - similarity_scores_pos) + torch.relu(1 + similarity_scores_neg))
        optimizer_question.zero_grad()
        
        loss.backward()
        optimizer_question.step()
        total_loss += loss.item()
        
    for batch in tqdm(dataloader_dev, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        
        question_input = tokenizer(batch["question"], return_tensors="pt", padding=True, truncation=True).to(device)
        positive_input = tokenizer(batch["positive_question"], return_tensors="pt", padding=True, truncation=True).to(device)
        negative_input = tokenizer(batch["negative_question"], return_tensors="pt", padding=True, truncation=True).to(device)
        
        question_embedding = question_model(**question_input).last_hidden_state.mean(dim=1)
        positive_embedding = question_model(**positive_input).last_hidden_state.mean(dim=1)
        negative_embedding = question_model(**negative_input).last_hidden_state.mean(dim=1)
        
        
        similarity_scores_pos=criterion(question_embedding, positive_embedding).mean()
        similarity_scores_neg=criterion(question_embedding, negative_embedding).mean()
        
        loss = torch.mean(torch.relu(1 - similarity_scores_pos) + torch.relu(1 + similarity_scores_neg))
        total_loss_dev += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}, Loss_dev: {total_loss_dev / len(dataloader_dev)}")
    os.makedirs('./model', exist_ok=True)
    torch.save(question_model.state_dict(), 'model/question_model_epoch'+str(epoch)+'.pth')

tokenizer.save_vocabulary('distilbert_tokenizer')

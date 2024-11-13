import os.path
import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Load positive and negative sentiment data
with open('1.txt', encoding='utf-8', errors='ignore') as f:
    pos_data = f.readlines()
with open('0.txt', encoding='utf-8', errors='ignore') as f:
    neg_data = f.readlines()

# Load prediction data
with open('merged_file.txt', encoding='utf-8', errors='ignore') as f:
    predict_data = f.readlines()
predict_index = []

for i, t_data in enumerate(predict_data):
    aa = t_data.strip('\n').strip('?').strip(' ').strip(']').strip('@').strip('#').split('\\')
    for j, a in enumerate(aa):
        a = a.strip('?').strip(' ').strip(']')
        if a == '':
            continuemain.py
        if a[0] == 'u' and len(a) != 5:
            del aa[j]
    t_data = '\\'.join(aa)
    t_data = t_data.replace('\\u\n', '').strip('\\').encode("utf-8").decode("unicode_escape").replace('\n', '').replace('???', '')
    t_data_list = t_data.split()
    text = ' '.join(t_data_list[1:])
    index = t_data_list[0]
    predict_index.append(int(index))
    predict_data[i] = text

pos_label = [1] * len(pos_data)
neg_label = [0] * len(neg_data)
all_data = pos_data + neg_data
all_label = pos_label + neg_label

# Split data into training and testing sets
x_train, x_test, train_label, test_label = train_test_split(
    all_data[:], all_label[:], test_size=0.1, stratify=all_label[:])

# Tokenize data using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=128)
predict_encoding = tokenizer(predict_data, truncation=True, padding=True, max_length=128)


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encoding, train_label)
test_dataset = ReviewDataset(test_encoding, test_label)
predict_dataset = ReviewDataset(predict_encoding, predict_index)

# Load a pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
predict_dataloader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

# Optimizer
optim = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)

def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 10 == 0):
            print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
            epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))

# Define the validation loop
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")
    return avg_val_accuracy, total_eval_loss / len(test_dataloader)

# Training and evaluation process
valid_loss = []
best_loss = 100000.0
best_epoch = 0
ckpts_path = './ckpts'
if not os.path.exists(ckpts_path):
    os.makedirs(ckpts_path)
for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    val_acc, val_loss = validation()
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
    torch.save(model, ckpts_path + '/' + 'model-{}.pth'.format(epoch))

model = torch.load(ckpts_path + '/' + 'model-{}.pth'.format(best_epoch))
model.eval()
index_list = []
pred_result_list = []
label_dict = {1: 'pos', 0: 'neg'}

# Make predictions
for batch in predict_dataloader:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    indexies = batch['labels']
    logits = logits.detach().cpu().numpy()
    indexies = indexies.to('cpu').numpy().tolist()
    preds_flat = np.argmax(logits, axis=1).flatten().tolist()
    index_list = index_list + indexies
    pred_result_list = pred_result_list + [label_dict[p] for p in preds_flat]

# Save
with open('predict_results_merged_file.txt', 'w') as f:
    for inx, pred in zip(index_list, pred_result_list):
        f.write(str(inx) + '\t' + pred + '\n')

import numpy as np
import pandas as pd

from string import punctuation
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

train_fpath = 'data/Constraint_English_Train.xlsx'
val_fpath = 'data/Constraint_English_Val.xlsx'
test_fpath = 'data/english_test_with_labels.xlsx'

# load raw data from xlsx files
def load_data():
    train = pd.read_excel(train_fpath)
    val = pd.read_excel(val_fpath)
    test = pd.read_excel(test_fpath)
    #print(train_raw_data)
    return train, val, test

# remove punctuation
def clean_text(file):
    texts = []
    word = ''
    for text in file['tweet']:
        for c in text.lower():
            if c not in punctuation:
                word += ''.join(c)
        texts.append(word)
        word = ''

    all_text = ' '.join(texts)
    words = all_text.split()
    #print(texts)
    #print(words)
    #print(len(words))
    return words, texts

def embedding_word(words, texts, file):
    'input: word list, news string'
    'output: text_ints - tokenized texts index list, encoded_labels - labels list'
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: index for index, word in enumerate(vocab, 1)}
    #print(len(vocab_to_int))

    text_ints = []
    for text in texts:
        text_ints.append([vocab_to_int[word] for word in text.split()])
        #print(text_ints)

    encoded_labels = np.array([1 if label == "real" else 0 for label in file['label']])
    #print(encoded_labels)
    return text_ints, encoded_labels

def pad_features(text_ints, seq_length):
    'input: seq_length - size of padded sequence'
    'output: features of text_ints, leftmost padded with 0 or truncated to size seq_length'
    features = np.zeros((len(text_ints), seq_length), dtype=int)

    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    #print(features[0])
    return features

def process_data(file, seq_length):
    words, texts = clean_text(file)
    text_ints, Y = embedding_word(words, texts, file)
    X = pad_features(text_ints, seq_length=seq_length)

    return X, Y

def batch_data(X, Y, batch_size=10, train_data=False):
    data = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(data, shuffle=train_data, batch_size=batch_size, num_workers=5)
    return loader


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2):
        super(LSTMModel, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

def train_LSTM():
    rnn = LSTMModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    epochs = 5
    for e in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        num_correct = 0
        batch_size = 10

        # train
        hidden = rnn.init_hidden(batch_size)
        rnn.train()
        for inputs, labels in train_loader:
            hidden = tuple([each.data for each in hidden])
            rnn.zero_grad()

            output, hidden = rnn(inputs, hidden)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size

        # validation
        val_hidden = rnn.init_hidden(batch_size)
        rnn.eval()
        for inputs, labels in test_loader:
            val_hidden = tuple([each.data for each in val_hidden])

            output, val_hidden = rnn(inputs, val_hidden)
            loss = criterion(output.squeeze(), labels.float())

            pred = torch.round(output.squeeze())  # rounds to 0 or 1
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy())
            num_correct += np.sum(correct)
            valid_loss += loss.item() * batch_size

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(test_loader.dataset)

        print("Loss: {:.6f}".format(train_loss),
              "Val Loss: {:.6f}".format(valid_loss),
              "Accuracy: {:.6f}".format(num_correct / len(test_loader.dataset)))


if __name__ == "__main__":
    train_raw_data, val_raw_data, test_raw_data = load_data()

    seq_length = 50
    X_train, Y_train = process_data(train_raw_data, seq_length)
    #X_val, Y_val = process_data(val_raw_data, seq_length)
    X_test, Y_test = process_data(test_raw_data, seq_length)
    #print(X_train.shape, X_val.shape, X_test.shape)

    train_loader = batch_data(X_train, Y_train, train_data=True)
    #val_loader = batch_data(X_val, Y_val)
    test_loader = batch_data(X_test, Y_test)

    vocab_size = (21663) + 1 # word tokens + 0 padding
    output_size = 1
    embedding_dim = 50
    hidden_dim = 128
    n_layers = 2

    train_LSTM()

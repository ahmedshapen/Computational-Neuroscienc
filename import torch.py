import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

text = ["I", "love", "deep", "learning"]

word_to_ix = {word: i for i, word in enumerate(set(text))}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(word_to_ix)


input_seq = text[:3]  
target = text[3]      


input_tensor = torch.tensor([word_to_ix[word] for word in input_seq], dtype=torch.long)
target_tensor = torch.tensor([word_to_ix[target]], dtype=torch.long)


input_tensor = input_tensor.view(3, 1)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        out = self.fc(out[-1])  
        return out

embed_size = 10
hidden_size = 20
model = SimpleRNN(vocab_size, embed_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")

with torch.no_grad():
    output = model(input_tensor)
    predicted_idx = torch.argmax(output).item()
    print("Predicted word:", ix_to_word[predicted_idx])

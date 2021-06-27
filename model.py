import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, dropout=0.5,
                            num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        captions = captions[:, :-1]
        captions = self.embed(captions)
        
        self.batch_size = captions.size(0)
        
        self.hidden = torch.zeros((self.num_layers, self.batch_size, self.hidden_size)) 
        self.cell = torch.zeros((self.num_layers, self.batch_size, self.hidden_size))
        
        self.hidden, self.cell = self.hidden.to(device), self.cell.to(device)
        
        features = features.unsqueeze(1)
        
        values = torch.cat((features, captions), dim=1)
        out, (self.hidden, self.cell) = self.lstm(values, (self.hidden, self.cell))
        out = self.fc(out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        
        #required vars
        self.batch_size = inputs.size(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        outs = []
        
        #lstm vars
        self.hidden = torch.zeros((self.num_layers, self.batch_size, self.hidden_size)).to(device)
        self.cell = torch.zeros((self.num_layers, self.batch_size, self.hidden_size)).to(device)
        
        while True:
            
            out, (self.hidden, self.cell) = self.lstm(inputs, (self.hidden, self.cell))
            out = self.fc(out)
            max_index = torch.argmax(out, dim=2)
            outs.append(max_index.squeeze().cpu().numpy().item())
            
            if(max_index == 1 or len(outs) >= max_len):
                break
                
            inputs = self.embed(max_index)
            
        return outs
 
        
import torch
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, num_classes = 1000):
        super(EncoderCNN,self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_classes, embed_size)
        )

    def forward(self, images):
        features = self.block1(images)
        features = self.block2(features)
        features = self.block3(features)
        features = self.block4(features)
        features = self.block5(features)
        features = torch.reshape(features, (features.size(0), -1))
        features = self.fc_layers(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        features = torch.unsqueeze(features, 0)
        embeddings = torch.cat((features, embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
class Image_Captioning(nn.Module):
    def __init__(self, num_classes, embed_size, hidden_size, vocab_size, num_layers):
        super(Image_Captioning, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size, num_classes)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_return(self, images, vocab, max_len = 50):
        with torch.no_grad():
            feature = torch.unsqueeze(self.encoderCNN(images), 0)       # generates features of shape (1, batch_size, embed_size)
            token = None                                                # We can start with providing the <SOS> token or nothing and let it predict the <SOS>. I chose to let it predict.
            predicted_captions = []
            count = list(range(len(images)))

            for _ in range(max_len):
                hiddens, token = self.decoderRNN.lstm(feature, token)         # using the LSTM layer to get hidden units of shape (1, batch_size, hidden_size), this is for predicting the next feature from the provided previous feature and token
                output = self.decoderRNN.linear(torch.squeeze(hiddens, 0))    # linear mapping the hidden units to vocab, to generate output of shape (batch_size, vocab_size)
                prediction = torch.argmax(output, 1)                          # take the neuron with highest probability among vocabulary neurons, to get prediction of shape (batch_size)
                feature = torch.unsqueeze(self.decoderRNN.embed(prediction), 0)   # setting the features equal to the embedding of the last predicted word

                for i in count:
                    word_idx = prediction[i].item()
                    if word_idx == vocab.string_to_index["<EOS>"]:
                        count.pop(i)
                    elif len(predicted_captions)<=len(prediction):
                        predicted_captions.append([vocab.index_to_string[word_idx]] if word_idx in vocab.index_to_string else ["<UNK>"])
                    else:
                        predicted_captions[i].append(vocab.index_to_string[word_idx] if word_idx in vocab.index_to_string else "<UNK>")

        return predicted_captions

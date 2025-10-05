import models 
import dataloader as get_data
import torch
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import itertools

img_folder = r"C:\Users\ASUS\Desktop\image_captions\Images"
caption_file = r"C:\Users\ASUS\Desktop\image_captions\captions.txt"

# Hyperparameters
embed_size = 256
hidden_size = 256
batch_size = 32
num_epochs = 5
learning_rate = 1e-4
num_layers = 10
num_classes = 1000
num_workers = 2

transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),]
)

# Grid of Hyperparameters for grid searching
hyperparameter_grid = {
    "learning_rate" : [1e-2, 1e-3, 1e-4],
    "batch_size" : [16, 32, 64],
}

def train_val_model(hyperparams, dataloader, vocab_size, vocab):
    train_loader = dataloader['train']
    val_loader = dataloader['val']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.Image_Captioning(
        num_classes= num_classes,
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
    ).to(device)

    # creaitng loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.string_to_index["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    # Training
    model.train()
    for epoch in tqdm(range(num_epochs)):
        avg_train_score = 0
        total_loss = 0
        count = 0
        print(f"\nEpoch {epoch + 1}:\n")
        for idx ,(imgs, captions) in enumerate(train_loader):
            score = 0
            count += 1
            imgs = imgs.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()

            outputs = model(imgs, captions[:-1])
            train_loss = loss_fn(torch.reshape(outputs, (-1, outputs.shape[2])), captions.reshape(-1))
            total_loss += train_loss

            predicted_captions = model.caption_return(imgs, vocab)
            captions = torch.transpose(captions, 0, 1)
            reference = []
            for i in range(len(captions)):

                '''creating the reference: It was not generating the caption as I was trying to access caption[i][j] from vocab.index_to_string
                but captions[i][j] is a singleton tensor, not an integer. I did not realise this until very late. What i wrote was:'''
                # reference.append([vocab.index_to_string[captions[i][j]] for j in range(len(captions[i])) if ((captions[i][j] in vocab.index_to_string) and (captions[i][j] != vocab.string_to_index["<PAD>"]))])

                # creating the correct reference here
                reference.append([vocab.index_to_string[captions[i][j].item()] for j in range(len(captions[i])) if ((captions[i][j].item() in vocab.index_to_string) and (captions[i][j].item() != vocab.string_to_index["<PAD>"]))])
                score += sentence_bleu([reference[i]], predicted_captions[i])

            score /= hyperparams['batch_size']
            avg_train_score += score
            if (count%50 == 0):
                print(f"Batch {count},\tLoss: {train_loss:.2f},\tAccuracy: {score*100:.2f}%\n")

            train_loss.backward()     #Backpropagation
            optimizer.step()

        avg_train_score /= len(train_loader)
        avg_loss = total_loss/len(train_loader)
        print(f"Avg Train Loss: {avg_loss:.2f},\tAverage Accuracy: {avg_train_score*100:.2f}%\n")

        torch.save(model.state_dict(), f"model_weights.pth")
        torch.save(optimizer.state_dict(), f"optimiser.pth")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (imgs, captions) in enumerate(val_loader):
                imgs = imgs.to(device)
                captions = captions.to(device)
                outputs = model(imgs, captions[:-1])
                loss = loss_fn(torch.reshape(outputs, (-1, outputs.shape[2])), captions.reshape(-1))
                val_loss += loss.item()

        val_loss = val_loss/len(val_loader)
        print(f"Val Loss: {val_loss}\n")

    return val_loss

def grid_search(hyperparameter_grid, get_loader):
    best_hyperparams = None
    best_val_loss = float('inf')

    keys, values = zip(*hyperparameter_grid.items())
    for v in itertools.product(*values):
        hyperparams = dict(zip(keys, v))
        print(f"\nEvaluating hyperparameters: {hyperparams}\n")

        dataloader, dataset = get_loader(
            img_folder=img_folder,
            caption_file=caption_file,
            transform=transform,
            val_split = 0.15,
            test_split=0.1,
            batch_size=hyperparams['batch_size'],
            num_workers= num_workers
        )
        vocabulary = dataset.vocab
        vocab_size = dataset.vocab.__len__()
        print(f"vocab_size = {vocab_size}\n")

        val_loss = train_val_model(hyperparams, dataloader, vocab_size, vocabulary)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparams = hyperparams

    print(f"\nBest Hyperparameters: {best_hyperparams}\n")
    return best_hyperparams

best_hyperparams = {}
best_hyperparams = grid_search(hyperparameter_grid, get_data.get_loader)
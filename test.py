import models 
import dataloader as get_data
import torch
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu
import train

img_folder = r"C:\Users\ASUS\Desktop\image_captions\Images"
caption_file = r"C:\Users\ASUS\Desktop\image_captions\captions.txt"

# Hyperparameters
embed_size = 256
hidden_size = 256
num_epochs = 2
num_layers = 10
num_classes = 1000
num_workers = 2
batch_size = 32

transform = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),]
)

dataloader, dataset = get_data.get_loader(
    img_folder=img_folder,
    caption_file=caption_file,
    transform=transform,
    val_split = 0.15,
    test_split=0.1,
    batch_size=(train.best_hyperparams['batch_size'] if train.best_hyperparams else batch_size),
    num_workers= num_workers
)
vocabulary = dataset.vocab
vocab_size = dataset.vocab.__len__()

def test(dataloader, vocab_size, vocab):
    test_loader = dataloader['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.Image_Captioning(
        num_classes= num_classes,
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
    ).to(device)

    if torch.load:
        model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    score = 0

    for idx ,(imgs, captions) in enumerate(test_loader):
        imgs = imgs.to(device)
        captions = torch.transpose(captions, 0, 1).to(device)
        predicted_captions = model.caption_return(imgs, vocab)
        reference = []

        for i in range(len(captions)):
            hypothesis = predicted_captions[i]
            reference.append([vocab.index_to_string[captions[i][j]] for j in range(len(captions[i])) if ((captions[i][j] in vocab.index_to_string) and (captions[i][j] != vocab.string_to_index["<PAD>"]))])
        score += sentence_bleu([reference[i]], hypothesis)
        img = transforms.ToPILImage(imgs[i])
        img.show()
        print(f"Expected Caption: {captions[i]}\nPredicted Caption: {predicted_captions[i]}\n\n")

    avg_score = score/len(test_loader)
    print(f"Accuracy: {score*100}%")
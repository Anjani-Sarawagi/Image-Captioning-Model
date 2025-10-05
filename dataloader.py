import os
import spacy
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

import spacy.cli
spacy.cli.download("en_core_web_sm")

spacy_eng = spacy.load("en_core_web_sm")
img_folder = r"C:\Users\ASUS\Desktop\image_captions\Images"
caption_file = r"C:\Users\ASUS\Desktop\image_captions\captions.txt"

# Preprocess caption for easier implementation
def caption_cleaner(captions):
    for i in range(len(captions)):
        caption = captions[i]
        caption = caption.lower().strip()           # lowercase all characters
        caption = caption.replace("\"", '')
        caption = caption.replace('[^a-z]' , '')    # removes all special characters
        caption = caption.replace('\s+', ' ')       # replaces multiple spaces by single space
        caption = f"<SOS> {caption} <EOS>"
        captions[i] = caption
    return captions

class vocab():
    def __init__(self, freq_thresh):
        self.string_to_index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index_to_string = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.freq_thresh = freq_thresh

    def __len__(self):
        return len(self.string_to_index)

    @staticmethod
    def tokenizer_eng(text):
        return [word.text for word in spacy_eng(text)]

    def create_vocab(self, captions):
        freq = {}
        idx = 4

        for sentence in captions:
            for word in self.tokenizer_eng(sentence):
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == self.freq_thresh:
                    self.string_to_index[word] = idx
                    self.index_to_string[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_txt = self.tokenizer_eng(text)
        return [
            self.string_to_index[token] if token in self.string_to_index else self.string_to_index["<UNK>"]
            for token in tokenized_txt
        ]

class dataset8k(Dataset):
    def __init__(self, img_folder, caption_file, transform = None, freq_thresh = 3):
        self.img_folder = img_folder
        self.dataset = pd.read_csv(caption_file)
        self.transform  = transform

        # Getting lists for image_ids and captions
        self.imgs = self.dataset["image"]
        self.captions = self.dataset["caption"]
        self.captions = caption_cleaner(list(self.captions))

        # Creating a vocabulary
        self.vocab = vocab(freq_thresh)
        self.vocab.create_vocab(list(self.captions))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_id = self.imgs[index]
        caption = self.captions[index]
        img = Image.open(os.path.join(self.img_folder, img_id)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = self.vocab.numericalize(caption)
        return img, torch.tensor(numericalized_caption)

class MyCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [torch.unsqueeze(item[0], 0) for item in batch]
        imgs = torch.cat(imgs, dim = 0)
        cap = [item[1] for item in batch]
        cap = pad_sequence(cap, batch_first = False, padding_value = self.pad_idx)

        return imgs, cap
    
def get_loader(
        img_folder,
        caption_file,
        transform,
        val_split,
        test_split,
        batch_size = 32,
        num_workers = 2,
        shuffle = True,
        pin_memory = False,
):
    dataset = dataset8k(img_folder, caption_file, transform = transform)
    pad_idx = dataset.vocab.string_to_index["<PAD>"]

    train_idx, val_test_idx = train_test_split(list(range(len(dataset))), test_size=(val_split + test_split))
    val_idx, test_idx = train_test_split(list(range(len(val_test_idx))), test_size=(test_split/(test_split + val_split)))

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn= MyCollate(pad_idx),
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn= MyCollate(pad_idx),
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn= MyCollate(pad_idx),
    )
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, dataset

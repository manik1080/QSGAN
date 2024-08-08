import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.manifold import TSNE
import numpy as np


class DatasetLoader(Dataset):
    def __init__(self, labels, label=0, transform=None):
        """
        Args:
            labels (pandas.DataFrame): DataFrame containing the labels and values.
            label (int): Label to filter by.
            transform (callable, optional): Optional transform to be applied
                on a sample
        """
        self.labels = labels
        self.transform = transform
        self.df = self.filter_by_label(label)

    def filter_by_label(self, label):
        df = self.labels[self.labels['Fraud'] == label]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        array = self.df.iloc[idx, 0]# / 16
        array = np.array(array)
        image = array.astype(np.float32).reshape(8, 8)
        #print(image.shape)

        cat_label = np.array(self.df.iloc[idx, 2])
        cat_label = torch.tensor(cat_label, dtype=torch.float).view(-1)

        if self.transform:
            image = self.transform(image)
            #print(image.shape)

        # Return image, label and categorical label
        return image, self.df.iloc[idx, 1], cat_label


class FinancialFraudDataset:
    def __init__(self, data, labels, image_size, n_components, batch_size=1):
        self.image_size = image_size
        self.batch_size = batch_size
        self.data = data
        self.tsne_data = None
        self.le = LabelEncoder()
        self.labels = np.array(self.le.fit_transform(labels))
        self.ohe = OneHotEncoder()
        self.labels_categorical = ohe.fit_transform(self.labels.reshape(-1, 1)).toarray()
        self.n_components = n_components

    def preprocess(self, n_components):
        corpus = np.array(self.data)
        tfidf_vectorizer = tfidf(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()
        print("Created TF-IDF matrix of shape: ", tfidf_matrix.shape)

        n_components = 64
        tsne = TSNE(n_components=self.n_components, method='exact', random_state=42)
        self.tsne_data = tsne.fit_transform(tfidf_matrix)

    def create_train_test_loaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        
        data = pd.DataFrame({'Filings': self.tsne_data.tolist(), 'Fraud': self.labels, 'Categorical': self.labels_categorical.tolist()})
        x_train, x_test, y_train, y_test = train_test_split(self.data['Filings'], self.data[['Fraud', 'Categorical']], test_size=0.2, random_state=42)

        train_dataset = DatasetLoader(labels=pd.concat([x_train, y_train], axis=1), transform=transform)
        test_dataset = DatasetLoader(labels=pd.concat([x_test, y_test], axis=1), transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        return train_loader, test_loader


if __name__ == '__main__':
    pass

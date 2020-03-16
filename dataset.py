from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, input_data):
        self.data, self.label = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

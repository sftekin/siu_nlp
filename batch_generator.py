from torch.utils.data import DataLoader
from dataset import TweetDataset


class BatchGenerator:
    def __init__(self, data_dict, set_names, **kwargs):
        self.data_dict = data_dict
        self.set_names = set_names

        self.batch_size = kwargs.get('batch_size', 16)
        self.num_works = kwargs.get('num_works', 4)
        self.shuffle = kwargs.get('shuffle', True)

        self.dataset_dict, self.dataloader_dict = self.__create_data()

    def generate(self, data_type):
        """
        :param data_type: can be 'test', 'train' and 'validation'
        :return: img tensor, label numpy_array
        """
        selected_loader = self.dataloader_dict[data_type]
        yield from selected_loader

    def __create_data(self):

        im_dataset = {i: TweetDataset(input_data=self.data_dict[i])
                      for i in self.set_names}

        im_loader = {i: DataLoader(im_dataset[i],
                                   batch_size=self.batch_size,
                                   shuffle=self.shuffle,
                                   num_workers=self.num_works,
                                   drop_last=True)
                     for i in self.set_names}
        return im_dataset, im_loader

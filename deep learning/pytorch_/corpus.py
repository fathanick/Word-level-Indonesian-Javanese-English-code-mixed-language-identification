import torch
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset


class Corpus(object):
    def __init__(self, input_path, min_word_freq, batch_size):
        # create list all of the fields
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)

        # create dataset
        self.train_data, self.val_data, self.test_data = SequenceTaggingDataset.splits(
            path = input_path,
            train = "train.tsv",
            validation = "val.tsv",
            test = "test.tsv",
            fields = (("word", self.word_field), ("tag", self.tag_field))
        )

        # convert fields to vocab list
        self.word_field.build_vocab(self.train_data.word, min_freq = min_word_freq)
        self.tag_field.build_vocab(self.train_data.tag)

        # create iterator for batch input
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets = (self.train_data, self.val_data, self.test_data),
            batch_size = batch_size
        )

        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]
from src.model.linear import DNN
from src.data import GenericDataset
import os


def test_size_of_dataset():
    features = 40
    os.environ["INPUT_FEATURES"] = str(features)
    dataset = GenericDataset()
    assert len(dataset[0][0]) == features

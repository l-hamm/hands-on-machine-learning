import numpy as np
from zlib import crc32

#Functions to split the dataset into train and test set

#Possible but not stable if data is updated (random function will put different items into train set on each run)
def split_train_set(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


#More stable way: check by id whether an item already have been in the train or test set
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier) & 0xffffffff < test_ratio * 2**32)

def split_train_test_by_id(data, test_ratio, id_column):
    ids=data[id_column]
    in_test_set=ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]



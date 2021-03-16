# Libraries and provided functions
import pandas as pd
import zipfile
from io import StringIO
import numpy as np
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm
import tarfile
from collections import namedtuple
import sys


def drop_unused_items(train, val, test):
    train_items = train.itemid.unique()
    val_items = val.itemid.unique()
    test_items = test.itemid.unique()

    droped_items = list((set(val_items) | set(test_items)) - set(train_items))
    if len(droped_items) == 0:
        return val, test
    val_mask = val.userid == droped_items[0]
    test_mask = test.userid == droped_items[0]
    for droped_item in droped_items:
        val_mask += val.itemid == droped_item
        test_mask += test.itemid == droped_item
    val = val[~val_mask]
    test = test[~test_mask]
    return val, test


def move_timestamps_to_end(x, max_order):
    new_order = x.groupby('timestamp', sort=True).grouper.group_info[0]
    x["timestamp"] = (max_order - new_order.max()) + new_order
    return x


def normalize_timestamp(x):
    x["timestamp"] = x.groupby(['timestamp', 'itemid'], sort=True).grouper.group_info[0]
    return x


def set_timestamp_length(x):
    x['length'] = len(x)
    return x


def to_coo(data):
    user_idx, item_idx, feedback = data['userid'], data['itemid'], data['rating']
    return user_idx, item_idx, feedback


def to_matrices(data):
    data = split_by_groups(data)

    data_max_order = data['timestamp'].max()
    data = data.groupby("index").apply(move_timestamps_to_end, data_max_order)

    data_shape = data[['index', 'timestamp']].max() + 1
    data_matrix = sp.sparse.csr_matrix((data['itemid'],
                                        (data['index'], data['timestamp'])),
                                    shape=data_shape, dtype=np.float64).todense()
    mask_matrix = sp.sparse.csr_matrix((np.ones(len(data)),
                                        (data['index'], data['timestamp'])),
                                    shape=data_shape, dtype=np.float64).todense()

    data_users = data.drop_duplicates(['index'])
    user_data_shape = data_users['index'].max() + 1
    user_vector = sp.sparse.csr_matrix((data_users['userid'],
                                        (data_users['index'], np.zeros(user_data_shape))),
                                        shape=(user_data_shape, 1), dtype=np.float64).todense()
    
    #print(user_vector)

    user_matrix = np.tile(user_vector, (1, data_shape[1]))

    #print(user_matrix)

    return data_matrix, mask_matrix, user_matrix


def train_val_test_split(data, frac):
    data = data.groupby("userid").apply(set_timestamp_length)
    max_time_stamp = data['length'] * frac
    timestamp = data['timestamp']
    data_train = data[timestamp < max_time_stamp * 0.9].groupby("userid").apply(normalize_timestamp)
    data_val = data[(0.9 * max_time_stamp <= timestamp) & (timestamp < 0.95 * max_time_stamp)]
    data_test = data[(0.95 * max_time_stamp <= timestamp) & (timestamp <= max_time_stamp)]

    data_val, data_test = drop_unused_items(data_train, data_val, data_test)
    data_val, data_test = data_val.groupby("userid").apply(normalize_timestamp), \
                        data_test.groupby("userid").apply(normalize_timestamp)

    return data_train, data_val, data_test


def split_by_groups(data, group_length=20):
    data["group"] = data['timestamp'] // group_length
    data["timestamp"] = data['timestamp'] % group_length
    data["index"] = data.groupby(['userid', 'group'], sort=False).grouper.group_info[0]
    return data


def get_prepared_data(data, frac=1):
    print("Normalizing indices to avoid gaps")
    # normalize indices to avoid gaps
    data['itemid'] = data.groupby('itemid', sort=False).grouper.group_info[0]
    data['userid'] = data.groupby('userid', sort=False).grouper.group_info[0]
    data = data.groupby("userid").apply(normalize_timestamp)

    # build sparse user-movie matrix
    print("Splitting into train, validation and test parts")

    data_train, data_val, data_test = train_val_test_split(data, frac)

    user_idx, item_idx, feedback = to_coo(data_train.copy())

    train_items, train_mask, train_users = to_matrices(data_train.copy())
    val_items, val_mask, val_users = to_matrices(data_val.copy())
    test_items, test_mask, test_users = to_matrices(data_test.copy())

    print('Done.')
    return (train_items, train_mask, train_users), \
        (val_items, val_mask, val_users), \
        (test_items, test_mask, test_users), \
        (user_idx, item_idx, feedback)



def get_data(data_path,dname,frac = 1):
    #f = 'lastfm_1K/userid-timestamp-artid-artname-traid-traname.tsv'
    lf_data = pd.read_csv(data_path + dname,sep = '\t')
    lf_data.columns = ['userid','itemid','timestamp']
    #lf_data['timestamp'] = pd.to_datetime(lf_data['timestamp'])
    lf_data['rating'] = np.ones(len(lf_data))
    lf_data = lf_data[['userid', 'timestamp', 'itemid', 'rating']]

    lf_data.itemid = lf_data.groupby('itemid', sort=False).grouper.group_info[0]
    itemid_max = lf_data.itemid.max()
    lf_data = lf_data[lf_data.itemid <= itemid_max * frac]

    lf_data = lf_data.drop_duplicates(['userid', 'timestamp', 'itemid'])
    return lf_data


if __name__ == '__main__':
    lf_data = get_data('Foursquare/','ratings.csv',1)
    #print(lf_data)
    #print(lf_data['itemid'].value_counts())
    
    (lf_train_items, lf_train_mask, lf_train_users), \
    (lf_val_items, lf_val_mask, lf_val_users), \
    (lf_test_items, lf_test_mask, lf_test_users), \
    (lf_train_user_idx, lf_train_item_idx, lf_train_feedback) = get_prepared_data(lf_data)
    
    data_path = 'Foursquare/'
    np.save(data_path + "lf_train_items",lf_train_items)
    np.save(data_path + 'lf_train_mask',lf_train_mask)
    np.save(data_path + 'lf_train_users',lf_train_users)
    np.save(data_path + 'lf_val_items',lf_val_items)
    np.save(data_path + 'lf_val_mask',lf_val_mask)
    np.save(data_path + 'lf_val_users',lf_val_users)
    np.save(data_path + 'lf_test_items',lf_test_items)
    np.save(data_path + 'lf_test_mask',lf_test_mask)
    np.save(data_path + 'lf_test_users',lf_test_users)
    np.save(data_path + 'lf_train_user_idx',lf_train_user_idx)
    np.save(data_path + 'lf_train_item_idx',lf_train_item_idx)
    np.save(data_path + 'lf_train_feedback',lf_train_feedback)
    '''
    print("-"*100)
    print("lf_train_items:", len(lf_train_items))
    print('lf_train_mask', len(lf_train_mask))
    print('lf_train_users', len(lf_train_users))
    print('lf_val_items', len(lf_val_items))
    print('lf_val_mask', len(lf_val_mask))
    print('lf_val_users', len(lf_val_users))
    print('lf_test_items', len(lf_test_items))
    print('lf_test_mask', len(lf_test_mask))
    print('lf_test_users', len(lf_test_users))
    print('lf_train_user_idx', lf_train_user_idx)
    print('lf_train_item_idx', lf_train_item_idx)
    print('lf_train_feedback', lf_train_feedback)
    '''
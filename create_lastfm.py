#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from scipy import sparse
import pandas as pd
import pickle

def save_weights_pkl(fname, weights):
    with open(fname, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)

def load_weights_pkl(fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f)
    return weights

parser = argparse.ArgumentParser("Data creation for VAE based collaborative filtering")
parser.add_argument('--dataset_name', type=str, default='lastfm-360k', help='dataset name', choices=['ml-20m', 'lastfm-360k'])
parser.add_argument('--out_data_dir', type=str, default='/path/to/dir', help='datadir')
parser.add_argument('--rating_file', type=str, default='/path/to/file', help='datadir')
parser.add_argument('--profile_file', type=str, default=None, help='datadir')
args = parser.parse_args()
print(args)

os.makedirs(args.out_data_dir, exist_ok=True)

raw_data_orig = pd.read_csv(args.rating_file, sep='\t', names=['userId', 'movieId', 'artist', 'rating'])
user_data_orig = pd.read_csv(args.profile_file, sep='\t', names=['userId', 'gender', 'age', 'country', 'signup'])

# gender: if unknown, replaced by 'na'
user_data_orig.gender.fillna('na', inplace=True)
user_data_orig.age.fillna(9999, inplace=True)
assert not user_data_orig.country.isnull().any()

# pd.get_dummies(user_data_orig.gender.head(), prefix = 'gender')

# age bins: [6–17], [18–21], [22–25], [26–30], [31–40], [41–50], [51–60], and [61–100].
bins = [-9999, 5, 17, 21, 25, 30, 40, 50, 60, 9998, 10000]
labels = list(range(len(bins)-1))
user_data_orig['age_bins'] = pd.cut(user_data_orig.age, bins=bins, labels=labels, include_lowest=False, right=True)
assert not user_data_orig.age_bins.isnull().any()

# country: top 16 in frequency and others
countries = user_data_orig.country.value_counts(dropna=False).rename_axis('country').reset_index(name='counts')
countries["country_rank"] = range(len(countries))
countries["country_rank"][countries["country_rank"] > 15] = 16

gender_onehot = pd.get_dummies(user_data_orig['gender'], prefix='g').head(20)
age_onehot = pd.get_dummies(user_data_orig['age_bins'], prefix='a').head(20)
countries_onehot = pd.get_dummies(countries["country_rank"], prefix='c').head(20)
user_data_orig = user_data_orig.merge(countries[['country', 'country_rank']], on='country', how='left')

b=raw_data_orig.userId.unique()
new_user_id = pd.DataFrame(b, columns=['userId']).reset_index().rename(columns={"index": "new_userId"})

new_movie_id = raw_data_orig['movieId'].drop_duplicates().reset_index().rename(columns={"index": "new_movieId"})
new_user_id = raw_data_orig['userId'].drop_duplicates().reset_index().rename(columns={"index": "new_userId"})

raw_data_orig = raw_data_orig.merge(user_data_orig, on='userId', how='left')
raw_data_full = raw_data_orig.merge(new_user_id, on='userId', how='left').merge(new_movie_id, on='movieId', how='left')
raw_data_full = raw_data_full.drop(labels=['userId', 'movieId', 'artist', 'signup'], axis=1).rename(columns={"new_userId": "userId", "new_movieId": "movieId"})

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users.
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who clicked on at least min_uc items
    # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

raw_data, user_activity, item_popularity = filter_triplets(raw_data_full, min_uc=40, min_sc=40)

raw_data["orig_index"] = raw_data.index.values

gender_onehot = pd.get_dummies(raw_data['gender'], prefix='g')
age_onehot = pd.get_dummies(raw_data['age_bins'], prefix='a')
countries_onehot = pd.get_dummies(raw_data["country_rank"], prefix='c')

raw_data = pd.concat([raw_data, gender_onehot, age_onehot, countries_onehot], axis=1)
raw_data = raw_data.drop(labels=['gender', 'age', 'country', 'age_bins', 'country_rank', 'orig_index'], axis=1)


sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] *
item_popularity.shape[0])
print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
      (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

unique_uid = user_activity.index

np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm] # shuffle

# create train/validation/test users
n_users = unique_uid.size
n_heldout_users = 10000

tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

raw_data["orig_index"] = raw_data.index.values

unique_sid = pd.unique(train_plays['movieId'])
unique_sid_df = pd.DataFrame(unique_sid, columns=["movieId"])
unique_sid_df["sid"] = range(unique_sid_df.shape[0])
unique_uid_df = pd.DataFrame(unique_uid, columns=["userId"])
unique_uid_df["uid"] = range(unique_uid_df.shape[0])

with open(os.path.join(args.out_data_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group) # n records for this user

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool') # array([False, False, False])
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

train_plays = train_plays.merge(unique_uid_df, on='userId', how='left')
train_plays = train_plays.merge(unique_sid_df, on='movieId', how='left')
train_plays["uid_fm0"] = train_plays["uid"] # add the same column as valid and test dataset

train_plays_profile = train_plays.drop_duplicates(subset="uid_fm0").filter(regex="^[ugac].*") # unique data for each user
assert train_plays_profile['uid'].shape[0] == n_users - n_heldout_users * 2

vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

vad_plays = vad_plays.merge(unique_uid_df, on='userId', how='left')
vad_plays = vad_plays.merge(unique_sid_df, on='movieId', how='left')

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)
vad_plays_tr.reset_index(drop=True, inplace=True)
vad_plays_te.reset_index(drop=True, inplace=True)

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

test_plays = test_plays.merge(unique_uid_df, on='userId', how='left')
test_plays = test_plays.merge(unique_sid_df, on='movieId', how='left')

test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
test_plays_tr.reset_index(drop=True, inplace=True)
test_plays_te.reset_index(drop=True, inplace=True)

train_data = train_plays.filter(items=["uid", "sid"], axis=1)
train_data.to_csv(os.path.join(args.out_data_dir, 'train.csv'), index=False)

vad_data_tr = vad_plays_tr.filter(items=["uid", "sid"], axis=1)
vad_data_tr.to_csv(os.path.join(args.out_data_dir, 'validation_tr.csv'), index=False)

vad_data_te = vad_plays_te.filter(items=["uid", "sid"], axis=1)
vad_data_te.to_csv(os.path.join(args.out_data_dir, 'validation_te.csv'), index=False)

start_idx = min(vad_plays_tr['uid'].min(), vad_plays_te['uid'].min())
end_idx = max(vad_plays_tr['uid'].max(), vad_plays_te['uid'].max())
vad_plays_tr['uid_fm0'] = vad_plays_tr['uid'] - start_idx
vad_plays_te['uid_fm0'] = vad_plays_te['uid'] - start_idx

vad_plays_profile = vad_plays_tr.drop_duplicates(subset="uid_fm0").filter(regex="^[ugac].*") # unique data for each user
assert vad_plays_profile['uid_fm0'].shape[0] == n_heldout_users

test_data_tr = test_plays_tr.filter(items=["uid", "sid"], axis=1)
test_data_tr.to_csv(os.path.join(args.out_data_dir, 'test_tr.csv'), index=False)

test_data_te = test_plays_te.filter(items=["uid", "sid"], axis=1)
test_data_te.to_csv(os.path.join(args.out_data_dir, 'test_te.csv'), index=False)

start_idx = min(test_plays_tr['uid'].min(), test_plays_te['uid'].min())
end_idx = max(test_plays_tr['uid'].max(), test_plays_te['uid'].max())
test_plays_tr['uid_fm0'] = test_plays_tr['uid'] - start_idx
test_plays_te['uid_fm0'] = test_plays_te['uid'] - start_idx

test_plays_profile = test_plays_tr.drop_duplicates(subset="uid_fm0").filter(regex="^[ugac].*") # unique data for each user
assert test_plays_profile['uid_fm0'].shape[0] == n_heldout_users

unique_sid = list()
with open(os.path.join(args.out_data_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)

def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data

train_data_csr = load_train_data(os.path.join(args.out_data_dir, 'train.csv'))

def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    assert pd.unique(tp_tr["uid"]).shape[0] == end_idx - start_idx + 1
    assert pd.unique(tp_te["uid"]).shape[0] == end_idx - start_idx + 1

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

vad_data_tr_csr, vad_data_te_csr = load_tr_te_data(os.path.join(args.out_data_dir, 'validation_tr.csv'), os.path.join(args.out_data_dir, 'validation_te.csv'))

test_data_tr_csr, test_data_te_csr = load_tr_te_data(os.path.join(args.out_data_dir, 'test_tr.csv'), os.path.join(args.out_data_dir, 'test_te.csv'))

fname = os.path.join(args.out_data_dir, 'data_csr.pkl')
datas = [train_data_csr, vad_data_tr_csr, vad_data_te_csr, test_data_tr_csr, test_data_te_csr, train_plays_profile, vad_plays_profile, test_plays_profile]
save_weights_pkl(fname, datas)

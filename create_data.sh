#!/usr/bin/env bash

PYTHON=python.exe

###############################
#
# create dateset for movielens-20m
# First, download the dataset at http://files.grouplens.org/datasets/movielens/ml-20m.zip


# data directory
DATASET=ml-20m
DATA_DIR=ml-20m
RATING_FILE_PATH=$DATA_DIR/ratings.csv

OUT_RATING_PATH=$DATA_DIR/ratings_35.csv

# select rows where the rating is greater than or equal to 3.5
cat $RATING_FILE_PATH | awk -v FS=$',' -v OFS=$',' '($3>3.5){print $1,$2,$3,$4}' > $OUT_RATING_PATH

PICKLE_OUT_DIR=$DATASET
$PYTHON create_ml.py --dataset_name $DATASET --out_data_dir $PICKLE_OUT_DIR --rating_file $OUT_RATING_PATH

###############################
#
# create dateset for lastfm-360k
# First, download the dataset(lastfm-dataset-360K.tar.gz) from http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html

DATASET=lastfm-360k
DATA_DIR=lastfm-dataset-360K
RATING_FILE_PATH=$DATA_DIR/usersha1-artmbid-artname-plays.tsv
USER_PROFILE_PATH=$DATA_DIR/usersha1-profile.tsv

OUT_RATING_PATH=$DATA_DIR/plays_50.tsv

# select rows where the number of plays is greater than or equal to 50
cat $RATING_FILE_PATH | awk -v FS=$'\t' -v OFS=$'\t' '($4>=50 && length($2) > 0 && length($1) == 40){print $1,$2,$3,$4}' > $OUT_RATING_PATH


PICKLE_OUT_DIR=$DATASET
$PYTHON create_lastfm.py --dataset_name $DATASET --out_data_dir $PICKLE_OUT_DIR --rating_file $OUT_RATING_PATH --profile_file $USER_PROFILE_PATH

exit 0

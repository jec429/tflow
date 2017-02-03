# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import numpy as np
import pandas as pd
import tensorflow as tf


COLUMNS = ["lead_lep_pt","lead_jet_pt","sublead_lep_pt","sublead_jet_pt","WR_mass","WR_event"]
LABEL_COLUMN = "label"
CONTINUOUS_COLUMNS = ["lead_lep_pt","lead_jet_pt","sublead_lep_pt","sublead_jet_pt","WR_mass"]
CATEGORICAL_COLUMNS = []

def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  return train_file_name, test_file_name


def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Continuous base columns.
  lead_lep_pt = tf.contrib.layers.real_valued_column("lead_lep_pt")
  lead_jet_pt = tf.contrib.layers.real_valued_column("lead_jet_pt")
  sublead_lep_pt = tf.contrib.layers.real_valued_column("sublead_lep_pt")
  sublead_jet_pt = tf.contrib.layers.real_valued_column("sublead_jet_pt")
  WR_mass = tf.contrib.layers.real_valued_column("WR_mass")
  
  # Transformations.
  #age_buckets = tf.contrib.layers.bucketized_column(age,boundaries=[18, 25, 30, 35, 40, 45,50, 55, 60, 65])

  # Wide columns and deep columns.
  wide_columns = [lead_lep_pt,lead_jet_pt,sublead_lep_pt,sublead_jet_pt,WR_mass
                  #tf.contrib.layers.crossed_column([lead_lep_pt, WR_mass],hash_bucket_size=int(1e4))
  ]
  
  deep_columns = [
  #    tf.contrib.layers.embedding_column(workclass, dimension=8),
  #    tf.contrib.layers.embedding_column(education, dimension=8),
  #    tf.contrib.layers.embedding_column(gender, dimension=8),
  #   tf.contrib.layers.embedding_column(relationship, dimension=8),
  #    tf.contrib.layers.embedding_column(native_country,
  #                                       dimension=8),
  #    tf.contrib.layers.embedding_column(occupation, dimension=8),
  #    age,
  #    education_num,
  #    capital_gain,
  #    capital_loss,
  #    hours_per_week,
    lead_lep_pt,lead_jet_pt,sublead_lep_pt,sublead_jet_pt,WR_mass
  ]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
    
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  #train_file_name, test_file_name = maybe_download(train_data, test_data)
  train_file_name, test_file_name = 'train.csv','test.csv'
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      #skiprows=1,
      engine="python")
  df_predict = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skipinitialspace=True,
      #skiprows=1,
      engine="python")
  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)
  df_predict = df_predict.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (
      df_train["WR_event"].apply(lambda x: "Background" in x)).astype(int)
  df_test[LABEL_COLUMN] = (
      df_test["WR_event"].apply(lambda x: "Background" in x)).astype(int)
  df_predict[LABEL_COLUMN] = (
      df_predict["WR_event"].apply(lambda x: "Background" in x)).astype(int)

  #print(df_train["WR_mass"])
  
  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))

  #for p in m.predict(input_fn=lambda: input_fn(df_predict)):
  #  print(p)

    
FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

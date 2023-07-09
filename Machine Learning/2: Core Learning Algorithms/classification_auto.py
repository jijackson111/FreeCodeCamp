from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np

# Dataset Inputs
tr_name = input('Insert filename of training data (e.g., train_data.csv): ')
tr_link = input('Insert .csv training dataset file link: ')
te_name = input('Insert filename of testing data (e.g., test_data.csv): ')
te_link = input('Insert .csv test dataset file link: ')

# Columns
CSV_COLUMN_NAMES = []
cn_i = input('List column names, using a comma then space to separate them: ')
cn_l = cn_i.split(', ')
for elem in cn_l:
    CSV_COLUMN_NAMES.append(elem)

# Possible results
POSSIBLE_RESULTS = []
pr_i = input('Now list the names of all possible results: ')
pr_l = pr_i.split(', ')
for el in pr_l:
    POSSIBLE_RESULTS.append(el)

# Training paths    
train_path = tf.keras.utils.get_file(tr_name, tr_link)
test_path = tf.keras.utils.get_file(te_name, te_link)
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
pop_val = input('Which column to leave out of the dataset: ')
train_y = train.pop(pop_val)
test_y = test.pop(pop_val)

# Gathering numbers for eval set
num_of_cols = len(CSV_COLUMN_NAMES) - 1
num_list = list(range(num_of_cols))
arrays = []
for num in num_list:
    app = []
    v1 = train.loc[0][num]
    v2 = train.loc[1][num]
    app.append(v1)
    app.append(v2)
    arrays.append(app)
    
# Evaluation set
def input_evaluation_set():
    new_cols = list(train.columns)
    p = 0
    features = {}
    while p < len(new_cols):
        features[new_cols[p]] = np.array(arrays[p])
        p += 1
    labels = np.array([2, 1])
    return features, labels

# Input function
bs = int(input('Enter desired Batch Size: '))

def input_fn(features, labels, training=True, batch_size=bs):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Feature Columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Building the model
ncl = int(input('Number of classes the model should choose between: '))
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=ncl)

# Train the Model.
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# Evaluation
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Get values for prediction dictionary
arrays2 = []
for num in num_list:
    app = []
    v1 = test.loc[0][num]
    v2 = test.loc[1][num]
    v3 = test.loc[2][num]
    app.append(v1)
    app.append(v2)
    app.append(v3)
    arrays.append(app)

# Get values for prediction dictionary
arrays2 = []
for num in num_list:
    app = []
    v1 = test.loc[0][num]
    v2 = test.loc[1][num]
    v3 = test.loc[2][num]
    app.append(v1)
    app.append(v2)
    app.append(v3)
    arrays2.append(app)

# Generate predictions from the model
expected = POSSIBLE_RESULTS
test_cols = list(test.columns)
predict_x = {}
e = 0
while e < num_of_cols:
    predict_x[test_cols[e]] = np.array(arrays2[e])
    e += 1

# Prediction function
def input_fn(features, batch_size=256):
    """An input function for prediction."""
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict_x))

# Display results

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(
        POSSIBLE_RESULTS[class_id], 100 * probability, expec))
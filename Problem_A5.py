# =======================================================================================
# PROBLEM A5
#
# Build and train a neural network model using the Sunspots.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from kaggle.com/robervalt/sunspots
#
# Desired MAE < 0.15 on the normalized dataset.
# ========================================================================================

import csv
import tensorflow as tf
import numpy as np
import urllib

# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_A5():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sunspots.csv'
    urllib.request.urlretrieve(data_url, 'sunspots.csv')

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
        sunspots.append(float(row[2]))  # Append the sunspots value to the sunspots list
        time_step.append(int(row[0]))  # Append the time step to the time_step list

    series = np.array(sunspots)  # Convert the sunspots list to a NumPy array

    # Normalization Function. DO NOT CHANGE THIS CODE
    min_val = np.min(series)
    max_val = np.max(series)
    series = (series - min_val) / (max_val - min_val)
    time = np.array(time_step)

    # DO NOT CHANGE THIS CODE
    split_time = 3000

    time_train = time[:split_time]  # Time steps for training data
    x_train = series[:split_time]  # Sunspot values for training data
    time_valid = time[split_time:]  # Time steps for validation data
    x_valid = series[split_time:]  # Sunspot values for validation data

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size,
                                 batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model with mean absolute error (MAE) as the metric
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6))

    # Train the model
    model.fit(train_set, epochs=100)

    return model

if __name__ == '__main__':
    model = solution_A5()
    model.save("model_A5.h5")

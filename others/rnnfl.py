import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_examples, test_examples = dataset['train'], dataset['test']

encoder = info.features['text'].encoder

print('Vocabulary size: {}'.format(encoder.vocab_size))
sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

BUFFER_SIZE = 10000
BATCH_SIZE = 32

train_set = (train_examples.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE ,padded_shapes=([None],[])))

test_set = (test_examples.padded_batch(BATCH_SIZE, padded_shapes=([None],[])))

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(encoder.vocab_size,64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(1))


callback = [
  tf.keras.callbacks.ModelCheckpoint(filepath="WBFL.h5",monitor="val_loss",verbose=0,save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0.1)
]
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              metrics=["accuracy"],
              )

history = model.fit(train_set, epochs=10,
                    validation_data=test_set,
                    validation_steps=30,
                    callbacks=callback)

test_loss, test_acc = model.evaluate(test_set)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
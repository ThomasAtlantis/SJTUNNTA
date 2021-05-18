import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np
import platform

random = np.random.RandomState(0)

print("Environment: TensorFlow %s, Python %s" % (tf.__version__, platform.python_version()))

_train_data = np.hstack([
    (np.load('data_hw2/train_data.npy') - 10) / 20, np.expand_dims(np.load('data_hw2/train_label.npy'), axis=1)])
_valid_data = np.hstack([
    (np.load('data_hw2/test_data.npy') - 10) / 20, np.expand_dims(np.load('data_hw2/test_label.npy'), axis=1)])

train_subjects, valid_subjects = 11, 4
sequence_length, labels_num = 60, 3
sample_per_subject = int(_train_data.shape[0] / train_subjects)
sample_per_video = np.array([238, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206])

sample_indexes = [0]
for i in range(sample_per_video.shape[0]):
    sample_indexes.append(sample_per_video[i] + sample_indexes[-1])
sample_indexes = np.stack([sample_indexes[:-1], sample_indexes[1:]], axis=-1)

train_data = [_train_data[l: r] for i in range(train_subjects) for l, r in (sample_indexes + sample_per_subject * i)]
train_data = [[d[sequence_length * i: sequence_length * (i + 1)] for i in range(d.shape[0] // sequence_length)] + [
    d[-sequence_length:]] for d in train_data]
train_data = np.array([_ for d in train_data for _ in d])

valid_data = [_valid_data[l: r] for i in range(valid_subjects) for l, r in (sample_indexes + sample_per_subject * i)]
valid_data = [[d[sequence_length * i: sequence_length * (i + 1)] for i in range(d.shape[0] // sequence_length)] + [
    d[-sequence_length:]] for d in valid_data]
valid_data = np.array([_ for d in valid_data for _ in d])

train_X, train_y = train_data[..., :310], to_categorical(train_data[:, 0, 310], num_classes=labels_num)
valid_X, valid_y = valid_data[..., :310], to_categorical(valid_data[:, 0, 310], num_classes=labels_num)

print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)


def ModelLSTM():
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(sequence_length, 310), activation='tanh', return_sequences=False))
    model.add(Dense(units=labels_num, activation='softmax'))
    return model


def ModelBase():
    model = Sequential()
    model.add(Dense(input_shape=(310,), units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=labels_num, activation='softmax'))
    return model


epoch_lstm, epoch_base = 300, 300

model_lstm = ModelLSTM()
model_lstm.compile(optimizer=Adam(2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.summary()
history_lstm = model_lstm.fit(train_X, train_y, epochs=epoch_lstm, batch_size=32, shuffle=True, validation_data=(valid_X, valid_y))

model_base = ModelBase()
model_base.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model_base.summary()
train_X, train_y = _train_data[::60, :310], to_categorical(_train_data[::60, 310], num_classes=labels_num)
valid_X, valid_y = _valid_data[::60, :310], to_categorical(_valid_data[::60, 310], num_classes=labels_num)
history_base = model_base.fit(train_X, train_y, epochs=epoch_base, batch_size=32, shuffle=True, validation_data=(valid_X, valid_y))

steps_lstm = np.arange(epoch_lstm) + 1
steps_base = np.arange(epoch_base) + 1
for i, keyword in enumerate(['loss', 'accuracy', 'val_accuracy']):
    plt.subplot(1, 3, i + 1)
    plt.title(keyword)
    plt.plot(steps_lstm, history_lstm.history[keyword], label="LSTM")
    plt.plot(steps_base, history_base.history[keyword], label="Baseline DNN")
    np.save(keyword + '_lstm.npy', np.array(history_lstm.history[keyword]))
    np.save(keyword + '_base.npy', np.array(history_base.history[keyword]))
    plt.legend()
    plt.grid()
plt.show()

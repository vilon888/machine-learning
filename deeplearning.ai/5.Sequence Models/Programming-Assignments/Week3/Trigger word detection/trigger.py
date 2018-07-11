import numpy as np
from td_utils import *

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1
    return segment_start, segment_end


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    isOverlap = False
    segment_start, segment_end = segment_time
    for start, end in previous_segments:
        if segment_start <= end and segment_end >= start:
            isOverlap = True

    return isOverlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)

    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """
    seg_end_y = int(segment_end_ms * Ty / 10000.0)
    y[0, seg_end_y+1:seg_end_y+51] = 1.0

    return y


def create_training_example(backgrounds, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    backgrounds -- a list of 10 second background audio recording, randomly choose one
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    np.random.seed(18)

    background = backgrounds[np.random.randint(0, len(backgrounds))]
    background -= 20
    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)

    random_indices = np.random.randint(len(activates), size=number_of_activates)

    random_activates = [activates[i] for i in random_indices]

    for activate in random_activates:
        background, segment_time = insert_audio_clip(background, activate, previous_segments)
        seg_start, seg_end = segment_time
        y = insert_ones(y, seg_end)

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for negative in random_negatives:
        background, _ = insert_audio_clip(background, negative, previous_segments)

    background = match_target_amplitude(background, -20.0)

    file_handle = background.export("train" + ".wav", format="wav")
    x = graph_spectrogram("train.wav")

    return x, y


def create_train_dev_set(train_number, dev_number):
    """
    Create train/dev examples, and put it into XY_train/X.npy, XY_train/Y.npy, XY_dev/X.npy, XY_dev/Y.npy,
    :param train_number: number of training examples
    :param dev_number: number of dev examples
    :return: train_x, train_y, dev_x, dev_y
    """
    train_x = np.zeros((train_number, n_freq, Tx))
    train_y = np.zeros((train_number, Ty))

    dev_x = np.zeros((dev_number, n_freq, Tx))
    dev_y = np.zeros((dev_number, Ty))

    for i in range(train_number):
        x, y = create_training_example(backgrounds, activates, negatives)
        train_x[i:] = x
        train_y[i:] = y[0]

    np.save('XY_train/X.npy', train_x)
    np.save('XY_train/Y.npy', train_y)

    # print('train_x\'s shape is: {}'.format(train_x.shape))

    for i in range(dev_number):
        x, y = create_training_example(backgrounds, activates, negatives)
        dev_x[i:] = x
        dev_y[i:] = y[0]

    np.save('XY_dev/X_dev.npy', dev_x)
    np.save('XY_dev/Y_dev.npy', dev_y)

    return train_x, train_y, dev_x, dev_y


def create_model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape=input_shape)

    X = Conv1D(196, 15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    model = Model(inputs=X_input, outputs=X)

    return model


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]

    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            consecutive_timesteps = 0

    audio_clip.export('chime_output.wav', format='wav')


m_train = 4000
m_dev = 25

Tx = 5511
n_freq = 101
Ty = 1375

activates, negatives, backgrounds = load_raw_audio()
# print('activates is : {}'.format(len(activates)))
# print('negatives is : {}'.format(len(negatives)))
# print('backgrounds is : {}'.format(len(backgrounds[0])))
#
# np.random.seed(5)
# audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
# audio_clip.export("insert_activate.wav", format="wav")
# print("Segment Time: ", segment_time)

# arr1 = insert_ones(np.zeros((1, Ty)), 9700)
# plt.plot(insert_ones(arr1, 4251)[0,:])
# plt.show()
# print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])


# x, y = create_training_example(backgrounds, activates, negatives)
# print(type(x))
# print(type(y))
# print(x.shape)
# print(y.shape)

# # creating train/dev data
# create_train_dev_set(500, 25)


X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")
X = np.transpose(X, axes=(0,2,1))
Y.shape = (Y.shape[0],Y.shape[1], 1)

X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")
X_dev = np.transpose(X_dev, axes=(0,2,1))
Y_dev.shape = (Y_dev.shape[0],Y_dev.shape[1], 1)

# model = model(input_shape=(Tx, n_freq))
#
# model.summary()

model = load_model('./models/tr_model.h5')

# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# model.fit(X, Y, batch_size=5, epochs=1)


loss, acc = model.evaluate(X_dev, Y_dev)
print('dev set accuracy =', acc)

chime_file = 'audio_examples/chime.wav'

filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)












































import os
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

from DataGenerator import DataGenerator
from utils import data_reader_from_npy

ROOT_PATH_TRAIN = r'data/WDF_embeddings_train_npy/'
ROOT_PATH_TEST = r'data/WDF_embeddings_test_npy/'

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2

HEAD_SIZE = 256
NUM_HEADS = 4
FF_DIM = 4
NUM_TRANSFORMER_BLOCKS = 4
MLP_UNITS = [128]
MLP_DROPOUT = 0.4
DROPOUT = 0.25

RUN_NUMBER = 4
MODEL_BEST_CHECKPOINT_PATH = f'./saved_models/WDF_Run{RUN_NUMBER:02d}_best.h5'
MODEL_SAVE_PATH = f'./saved_models/WDF_TimeSeries_Run{RUN_NUMBER:02d}'


def load_data(data_root_dir, which_set='train'):
    x, y = None, None
    n_files = len(os.listdir(data_root_dir))
    print(f'\nLoading Data: {which_set}')

    for i in tqdm(range(1, n_files + 1)):
        npy_filepath = os.path.join(data_root_dir, f'WDF_embeddings_{which_set}_{i}.npy')
        x_temp, y_temp = data_reader_from_npy(npy_filepath)
        if x is None and y is None:
            x, y = x_temp, y_temp
        else:
            x = np.vstack([x, x_temp])
            y = np.vstack([y, y_temp])
    y = y[:, 0]

    print(f'Data: {which_set} | shapes (x, y): {x.shape}, {y.shape}\n')

    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    return x, y


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.):
    """
    Build the model

    Our model processes a tensor of shape `(batch size, sequence length, features)`,
    where `sequence length` is the number of time steps and `features` is each input
    timeseries.

    You can replace your classification RNN layers with this one: the
    inputs are fully compatible!

    We include residual connections, layer normalization, and dropout.
    The resulting layer can be stacked multiple times.

    The projection layers are implemented through `keras.layers.Conv1D`.
    """

    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, n_classes, dropout=0.0,
                mlp_dropout=0.0):
    """
    The main part of our model is now complete. We can stack multiple of those
    `transformer_encoder` blocks and we can also proceed to add the final
    Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
    layers, we need to reduce the output tensor of the `TransformerEncoder` part of
    our model down to a vector of features for each data point in the current
    batch. A common way to achieve this is to use a pooling layer. For
    this example, a `GlobalAveragePooling1D` layer is sufficient.
    """

    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Masking(mask_value=0., input_shape=input_shape)(x)  # Mask out zero paddings
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    # outputs = layers.Dense(n_classes, activation="softmax")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def main():
    if os.path.exists(MODEL_BEST_CHECKPOINT_PATH):
        print('Model Checkpoint Already Exists. Update the Run number or choose different path.')
        return

    # Generators
    training_generator = DataGenerator(data_path=ROOT_PATH_TRAIN, which_set='train', npy_prefix='WDF_embeddings')
    x_test, y_test = load_data(ROOT_PATH_TEST, which_set='test')
    # print('Test data shape: ', x_test.shape, y_test.shape)

    # for i, (x, y) in enumerate(training_generator):
    #     print(i, x.shape, y.shape)

    input_shape = x_test.shape[1:]
    # n_classes = len(np.unique(y_train))

    model = build_model(
        input_shape,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_units=MLP_UNITS,
        n_classes=2,
        mlp_dropout=MLP_DROPOUT,
        dropout=DROPOUT,
    )

    # model.compile(
    #     loss="sparse_categorical_crossentropy",
    #     optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    #     metrics=["sparse_categorical_accuracy"],  # keras.metrics.AUC(), F1_score
    # )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],  # keras.metrics.AUC(), F1_score
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_BEST_CHECKPOINT_PATH, save_best_only=True)
    ]

    # model.fit(x_train, y_train, validation_split=VALIDATION_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE,
    #           callbacks=callbacks)

    model.fit_generator(generator=training_generator, validation_data=(x_test, y_test), epochs=EPOCHS,
                        callbacks=callbacks)

    model.evaluate(x_test, y_test, verbose=1)

    model.save(MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()

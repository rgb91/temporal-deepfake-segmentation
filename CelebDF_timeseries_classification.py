import os
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

from DataGenerator import DataGenerator
from utils import load_data_single_npy, load_data_multiple_npy

DATA_HOME = r'/mnt/sanjay/'
ROOT_PATH_TRAIN = os.path.join(DATA_HOME, r'CelebDF_embeddings_Method_B_train_npy/')
ROOT_PATH_TEST = os.path.join(DATA_HOME, r'CelebDF_embeddings_Method_B_test_npy/')

EPOCHS = 100
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
MODEL_BEST_CHECKPOINT_PATH = f'./saved_models/CelebDF_Method_B_Run{RUN_NUMBER:02d}_best.h5'
MODEL_SAVE_PATH = f'./saved_models/CelebDF_Method_B_TimeSeries_Run{RUN_NUMBER:02d}'


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
    # Generators
    # CelebDF_embeddings_Method_B
    training_generator = DataGenerator(data_path=ROOT_PATH_TRAIN, which_set='train',
                                       npy_prefix='CelebDF_embeddings_Method_B')
    x_test, y_test = load_data_multiple_npy(in_dir=ROOT_PATH_TEST, filename_prefix='CelebDF_embeddings_Method_B_test')

    if os.path.exists(MODEL_BEST_CHECKPOINT_PATH):
        print('Model Checkpoint Already Exists. Update the Run number or choose different path.')
        return

    input_shape = x_test.shape[1:]

    model = build_model(input_shape, head_size=HEAD_SIZE, num_heads=NUM_HEADS, ff_dim=FF_DIM,
                        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS, mlp_units=MLP_UNITS, n_classes=2,
                        mlp_dropout=MLP_DROPOUT, dropout=DROPOUT)
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
    # model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_BEST_CHECKPOINT_PATH, save_best_only=True)
    ]
    model.fit_generator(
        generator=training_generator, validation_data=(x_test, y_test), epochs=EPOCHS,
        callbacks=callbacks,
    )
    model.evaluate(x_test, y_test, verbose=1)

    model.save(MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
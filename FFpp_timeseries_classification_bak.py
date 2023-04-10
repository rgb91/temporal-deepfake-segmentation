from collections import Counter
from utils import load_data_single_npy
import os
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K


def load_data(root_path):
    x_train, y_train = load_data_single_npy(os.path.join(root_path, 'FFpp_embeddings_Method_A_train.npy'))
    y_train = y_train[:, 0]

    x_test, y_test = load_data_single_npy(os.path.join(root_path, 'FFpp_embeddings_Method_A_val.npy'))
    y_test = y_test[:, 0]

    print(x_train.shape, y_train.shape)
    # print(x_val.shape, y_val.shape)

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    return x_train, x_test, y_train, y_test


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.):
    """
    ## Build the model

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
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    """
    Train and evaluate

    Conclusions

    In about 110-120 epochs (25s each on Colab), the model reaches a training
    accuracy of ~0.95, validation accuracy of ~84 and a testing
    accuracy of ~85, without hyperparameter tuning. And that is for a model
    with less than 100k parameters. Of course, parameter count and accuracy could be
    improved by a hyperparameter search and a more sophisticated learning rate
    schedule, or a different optimizer.

    You can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/timeseries_transformer_classification)
    and try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/timeseries_transformer_classification).

    """
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    ROOT_PATH = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings'

    RUN_NUMBER = 6
    MODEL_BEST_CHECKPOINT_PATH = f'./saved_models/FFpp_Timeseries_Run{RUN_NUMBER:02d}_best.h5'
    MODEL_SAVE_PATH = f'./saved_models/FFpp_TimeSeries_Run{RUN_NUMBER:02d}'

    if os.path.exists(MODEL_BEST_CHECKPOINT_PATH):
        print('Model Checkpoint Already Exists. Update the Run number or choose different path.')
        exit(0)

    x_train, x_test, y_train, y_test = load_data(root_path=ROOT_PATH)

    input_shape = x_train.shape[1:]
    n_classes = len(np.unique(y_train))

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        n_classes=n_classes,
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_BEST_CHECKPOINT_PATH, save_best_only=True)
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

    model.save(MODEL_SAVE_PATH)
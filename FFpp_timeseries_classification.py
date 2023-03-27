import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from DataGenerator import DataGenerator
from utils import load_data_multiple_npy


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
    EPOCHS = 400
    N_CLASSES = 6
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4

    DATA_ROOT = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings'
    DATA_ROOT_TRAIN = os.path.join(DATA_ROOT, 'FFpp_train_embeddings_npy_batches')
    DATA_ROOT_TEST = os.path.join(DATA_ROOT, 'FFpp_test_embeddings_npy_batches')

    RUN_NUMBER = 9
    MODEL_BEST_CHECKPOINT_PATH = f'./saved_models/FFpp_TimeSeries_Run{RUN_NUMBER:02d}_best.h5'
    MODEL_SAVE_PATH = f'./saved_models/FFpp_TimeSeries_Run{RUN_NUMBER:02d}'

    if os.path.exists(MODEL_BEST_CHECKPOINT_PATH):
        print('Model Checkpoint Already Exists. Update the Run number or choose different path.')
        exit(0)

    training_generator = DataGenerator(data_path=DATA_ROOT_TRAIN, npy_prefix='FFpp_train_embeddings')
    x_test, y_test = load_data_multiple_npy(in_dir=DATA_ROOT_TEST, filename_prefix='FFpp_test_embeddings')

    input_shape = x_test.shape[1:]

    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128],
                        n_classes=N_CLASSES, mlp_dropout=0.4, dropout=0.25)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [
        # keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_BEST_CHECKPOINT_PATH, save_best_only=True)
    ]

    model.fit_generator(
        generator=training_generator, validation_data=(x_test, y_test), epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.evaluate(x_test, y_test, verbose=1)

    model.save(MODEL_SAVE_PATH)
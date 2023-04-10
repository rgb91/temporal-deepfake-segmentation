import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from DataGenerator import DataGenerator
from utils import load_data_multiple_npy


def build_model(input_shape, mlp_units, mlp_dropout, n_classes=2):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = layers.Masking(mask_value=0., input_shape=input_shape)(x)  # Mask out zero paddings

    # x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Flatten()(x)
    # print('\n\n\n\n', x.shape, '\n\n\n\n')
    # exit(1)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    # outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    EPOCHS = 200
    N_CLASSES = 2
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    PATIENCE = 10

    DATA_ROOT = r'/data/PROJECT FILES/DFD_Embeddings/FFpp_embeddings'
    DATA_ROOT_TRAIN = os.path.join(DATA_ROOT, 'FFpp_train_embeddings_npy_batches_5steps_binary')
    DATA_ROOT_TEST = os.path.join(DATA_ROOT, 'FFpp_test_embeddings_npy_batches_5steps_binary')

    BASELINE_RUN_NUM = 1
    MODEL_BEST_CHECKPOINT_PATH = f'./saved_models/FFpp_Baseline_Run{BASELINE_RUN_NUM:02d}_best_epoch_{epoch}.h5'
    MODEL_SAVE_PATH = f'./saved_models/FFpp_Baseline_Run{BASELINE_RUN_NUM:02d}'

    if os.path.exists(MODEL_BEST_CHECKPOINT_PATH):
        print('Model Checkpoint Already Exists. Update the Run number or choose different path.')
        exit(0)

    training_generator = DataGenerator(data_path=DATA_ROOT_TRAIN, npy_prefix='FFpp_train_embeddings',
                                       n_classes=N_CLASSES)
    validation_generator = DataGenerator(data_path=DATA_ROOT_TEST, npy_prefix='FFpp_test_embeddings',
                                         n_classes=N_CLASSES)
    # _path = os.path.join(DATA_ROOT, 'FFpp_test_embeddings_npy_batches_combined')
    # x_test, y_test = np.load(os.path.join(_path, 'x_test_5steps_binary.npy')), np.load(
    #     os.path.join(_path, 'y_test_5steps_binary.npy'))
    # x_test, y_test = load_data_multiple_npy(in_dir=DATA_ROOT_TEST, filename_prefix='FFpp_test_embeddings')
    # np.save('x_test_5steps_binary.npy', x_test)
    # np.save('y_test_5steps_binary.npy', y_test)

    # input_shape = x_test.shape[1:]
    input_shape = (5, 768)

    model = build_model(input_shape, mlp_units=[2048], mlp_dropout=0.4, n_classes=N_CLASSES)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )
    # model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_BEST_CHECKPOINT_PATH, save_best_only=False)
    ]
    # model.fit_generator(
    #     generator=training_generator, validation_data=(x_test, y_test), epochs=EPOCHS,
    #     callbacks=callbacks,
    # )
    model.fit(training_generator, validation_data=validation_generator, epochs=EPOCHS, callbacks=callbacks)
    # model.evaluate(x_test, y_test, verbose=1)
    # model.save(MODEL_SAVE_PATH)
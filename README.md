# Time Series Classification using Transformers

Machine: ```body.d2.comp.nus.edu.sg```

Conda Environment: ```dfl2```

**Directory Structure**:
```
DeepFakeDetectionTransformer
| data
| ---- CelebDF_embeddings_Method_B_train (contains CSV)
| ---- CelebDF_embeddings_Method_B_test (contains CSV)
| ---- CelebDF_embeddings_Method_B_train_npy
| ---- CelebDF_embeddings_Method_B_test_npy
| ---- FFpp_embeddings_Method_A_train_npy
| ---- ...
| saved_models
| ---- CelebDF_Method_B_Run01_best.h5
| ---- CelebDF_Method_B_Run02_best.h5
| ---- ...
| CelebDF_dataset_process.py
| CelebDF_evaluate.py
| CelebDF_prepare_data.py
| CelebDF_timeseries_classification.py
| FFpp_dataset_process.py
| FFpp_evaluate.py
| FFpp_prepare_data.py
| FFpp_timeseries_classification.py
| ...
```

**Run sequence**:

1. `CelebDF_dataset_process.py`: To process raw dataset. It converts videos to cropped (only face) frames.
2. `CelebDF_prepare_data.py`: To prepare data for training. Two stages: 1. It takes the embeddings for each video to 
convert to CSV files. 2. Convert the CSV files to NPY files. Each npy file contains 64 (`batch_size`) video segments. 
Each video segment contains 500 timesteps (frames). Any particular video can have one or more such segments.  
3. `CelebDF_time_series.py`: To train and save model. Uses `DataGenerator.py` to load data from the NPY files. 
4. `CelebDF_evaluate.py`: To evaluate and output metrics such as AUC, F1 scores. It takes a saved model, gets the 
predictions from the model for the test data and calculates these metrics. 

**References**:
1. https://keras.io/examples/timeseries/timeseries_transformer_classification/
2. ```tf.config.list_physical_devices('GPU')```
3. If you are using a training script you can simply set it in the command line before invoking the script ```CUDA_VISIBLE_DEVICES=1 python train.py```. Ref: https://stackoverflow.com/questions/53533974/how-do-i-get-keras-to-train-a-model-on-a-specific-gpu
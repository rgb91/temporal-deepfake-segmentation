# Time Series Classification using Transformers

Machine: ```body.d2.comp.nus.edu.sg```

Conda Environment: ```dfl2```

**References**:
1. https://keras.io/examples/timeseries/timeseries_transformer_classification/
2. ```tf.config.list_physical_devices('GPU')```
3. If you are using a training script you can simply set it in the command line before invoking the script ```CUDA_VISIBLE_DEVICES=1 python train.py```. Ref: https://stackoverflow.com/questions/53533974/how-do-i-get-keras-to-train-a-model-on-a-specific-gpu
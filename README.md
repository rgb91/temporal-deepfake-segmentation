# Undercover Deepfakes: Detecting Fake Segments in Videos

Repository to share the models and weights from the paper: [https://arxiv.org/pdf/xxxx.xxxx](https://arxiv.org/pdf/x)

## Abstract
> The recent renaissance in generative models, driven primarily by the advent of diffusion models and iterative improvement in GAN methods, has enabled many creative applications. However, each advancement is also accompanied by a rise in the potential for misuse. In the arena of deepfake generation this is a key societal issue. In particular, the ability to modify segments of videos using such generative techniques creates a new paradigm of deepfakes which are mostly real videos altered slightly to distort the truth. Current deepfake detection methods in the academic literature are not evaluated on this paradigm. In this paper, we present a deepfake detection method able to address this issue by performing both frame and video level deepfake prediction. To facilitate testing our method we create a new benchmark dataset where videos have both real and fake frame sequences. Our method utilizes the Vision Transformer, Scaling and Shifting pretraining and Timeseries Transformer to temporally segment videos to help facilitate the interpretation of possible deepfakes. Extensive experiments on a variety of deepfake generation methods show excellent results on temporal segmentation and classical video level predictions as well. In particular, the paradigm we introduce will form a powerful tool for the moderation of deepfakes, where human oversight can be better targeted to the parts of videos suspected of being deepfakes.

## Model Architecture
<!-- ![Model Architecture and Pipeline](/assets/architecture.png) -->
<img src="./assets/architecture.png" width=760>

## Results
### Temporal Segmentation
<!-- ![Temporal Segmentation Results](/assets/table-results-temporal.png) -->
<img src="./assets/table-results-temporal.png" width=760>

### Video Level Classification
<!-- ![Video Level Classification](/assets/table-results-video-level.png) -->
<img src="./assets/table-results-video-level.png" width=370>

## Model Weights
TBA
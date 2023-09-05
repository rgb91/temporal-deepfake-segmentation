# Undercover Deepfakes: Detecting Fake Segments in Videos

Accepted at [DFAD Workshop](https://ailb-web.ing.unimore.it/dfad2023/) in ICCV 2023: [[arXiv](https://arxiv.org/abs/2305.06564)|[pdf](https://arxiv.org/pdf/2305.06564.pdf)]

<!-- ![Header](/assets/intro_header.png) -->
<img src="./assets/intro_header.png" width=480>

## Abstract
The recent renaissance in generative models, driven primarily by the advent of diffusion models and iterative improvement in GAN methods, has enabled many creative applications. However, each advancement is also accompanied by a rise in the potential for misuse. In the arena of the deepfake generation, this is a key societal issue. In particular, the ability to modify segments of videos using such generative techniques creates a new paradigm of deepfakes which are mostly real videos altered slightly to distort the truth. This paradigm has been under-explored by the current deepfake detection methods in the academic literature. In this paper, we present a deepfake detection method that can address this issue by performing deepfake prediction at the frame and video levels. To facilitate testing our method, we prepared a new benchmark dataset where videos have both real and fake frame sequences with very subtle transitions. We provide a benchmark on the proposed dataset with our detection method which utilizes the Vision Transformer based on Scaling and Shifting to learn spatial features, and a Timeseries Transformer to learn temporal features of the videos to help facilitate the interpretation of possible deepfakes. Extensive experiments on a variety of deepfake generation methods show excellent results by the proposed method on temporal segmentation and classical video-level predictions as well. In particular, the paradigm we address will form a powerful tool for the moderation of deepfakes, where human oversight can be better targeted to the parts of videos suspected of being deepfakes.


## Temporal Dataset
Temporal dataset is prepared based on the FaceForensics++ (FF++) dataset. We publish the start and end frame number of the fake segment(s) in the CSV files in [temporal_dataset](temporal_dataset/) folder.
<!-- ![Temporal dataset](/assets/temporal_dataset.png) -->
<img src="./assets/temporal_dataset.png" width=760>
Manually selected fake segments where transition from real to fake frames and vice versa are very subtle.

## Model Architecture

We leverage Parameter Efficient Fine-Tuning (PEFT) to build an efficient transformer-based architecture that achieves results comparable to and outperforming SOTA methods.
<!-- ![Model Architecture and Pipeline](/assets/architecture.png) -->
<img src="./assets/architecture.png" width=760>


## Results
### Temporal Segmentation
<!-- ![Temporal Segmentation Results](/assets/table-results-temporal.png) -->
<img src="./assets/table-results-temporal.png" width=760>
On the subset with subtle transitions between real and fake frames.


<!-- ![Temporal Segmentation Results](/assets/table-results-temporal.png) -->
<img src="./assets/table-results-temporal.png" width=760>
On the subset with random (not subtle) transitions between real and fake frames.


### Video Level Classification
<!-- ![Video Level Classification](/assets/table-results-video-level.png) -->
<img src="./assets/table-results-video-level.png" width=370>


## Model Weights
Fine-tuned ViT model weights can be found [here](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=kMEoAeb6PUsHySXx7Ogw11282382393&browser=true&filename=checkpoint_best.pth.tar). 
ViT-embeddings for FF+ dataset can be downloaded [here](https://mediaflux.researchsoftware.unimelb.edu.au:443/mflux/share.mfjp?_token=TH4gLTKIH4bbwNECwkug11282382497&browser=true&filename=ff%2B_2_class_emb.zip).

We also thank the authors of the [SSF](https://github.com/dongzelian/SSF) for providing their source code.

## Cite this paper
    @misc{saha2023undercover,
          title={Undercover Deepfakes: Detecting Fake Segments in Videos},
          author={Sanjay Saha and Rashindrie Perera and Sachith Seneviratne and Tamasha Malepathirana and Sanka Rasnayaka and Deshani Geethika and Terence Sim and Saman Halgamuge},
          year={2023},
          eprint={2305.06564},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

# Learning in the Frequency Domain

## Highlights
* We propose a method of learning in the frequency domain (using DCT coefficients as input), which requires little modification to the existing CNN models that take RGB input.
* We show that learning in the frequency domain better preserves image information in the pre-processing stage than the conventional spatial downsampling approach.
* We propose a learning-based dynamic channel selection method to identify the trivial frequency components for static removal during inference. Experiment results on ResNet-50 show that one can prune up to $87.5\%$ of the frequency channels using the proposed channel selection method with no or little accuracy degradation in the ImageNet classification task.
* To the best of our knowledge, this is the first work that explores learning in the frequency domain for high-level vision tasks, such as object detection and instance segmentation.

Please refer to the [image classfication](classification) and [instance segmentation](segmentation) sections for more details.
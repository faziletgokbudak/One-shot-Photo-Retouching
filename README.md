## One-shot Detail Retouching [[Paper](https://faziletgokbudak.github.io/publications/cvmp23_4.pdf)] [[Project website](https://faziletgokbudak.github.io/projects/one-shot/)]

<!-- ## One-shot Detail Retouching with Patch Space Neural Field based Transformation Blending 
 -->
* Tensorflow implementation of *One-shot Detail Retouching with Patch Space Neural Transformation Blending*, which is accepted by ACM SIGGRAPH European Conference on Visual Media Production, 2023.
* This repo contains the main code.

### Required packages with suggested versions:
```
python==3.8
numpy==1.19.5
tensorflow==2.6.2
tensorflow-probability==0.12.2
opencv-contrib-python==4.6.0
```

### Useful Arguments:
```
[--input_path]          # Path to the 'before' image.
[--output_path]         # Path to the 'after' image.
[--test_path]           # Path to the input image.
[--test_output_path]    # Path to the output image, specified by the user.
[--model path]          # Path to the saved models.
[--num_matrices]        # Number of transformation matrices to be blended.
[--num_mlp]             # Number of MLPs to be used. For the latest version of our technique, keep it 1.
[--patch_size]          # Size of image patches to be processed.
[--laplacian_level]     # Number of Laplacian levels for frequency decomposition.
```

### Training:

```
python main.py --input_path=/DT_dataset/UM/train_images/Before.png --output_path=/DT_dataset/UM/train_labels/BeforeAfter_UM.png
```

### Testing:

```
python test.py --test_path=/DT_dataset/UM/test_images/Before.png --test_output_path=/output/output1.png --output_path=/DT_dataset/UM/train_labels/BeforeAfter_UM.png
```

### Results:
<p align="center">
  <img src='/output/Qualitative_zoomed.png'/><br/>
  <br/>Qualitative comparisons with state-of-the-art methods on different types of images<br/>
</p>

## Citation
If you find our work relevant to your research, please cite:
```
@inproceedings{10.1145/3626495.3626499,
author = {Gokbudak, Fazilet and Oztireli, A. Cengiz},
title = {One-Shot Detail Retouching with Patch Space Neural Transformation Blending},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626495.3626499},
doi = {10.1145/3626495.3626499},
series = {CVMP '23}
}
```

## One-shot Detail Retouching [[Paper](https://dl.acm.org/doi/pdf/10.1145/3626495.3626499)] [[Project website](https://faziletgokbudak.github.io/DetailRetouching/)]

<!-- ## One-shot Detail Retouching with Patch Space Neural Field based Transformation Blending 
 -->
* Tensorflow implementation of *One-shot Detail Retouching with Patch Space Neural Transformation Blending*, published at ACM SIGGRAPH European Conference on Visual Media Production, 2023.
* This repo contains the main code.

### Requirements:

To test our method, you first need to install the required packages, ideally in a virtual environment. I also recommend using python>=3.8. To install the packages, you can run the following:

```
pip install -r requirements.txt
```

### Useful Arguments:
```
[--input_path]                     # Path to a 'before' image.
[--output_path]                    # Path to an 'after' image.
[--test_path]                      # Path to an 'input' image.
[--test_output_path]               # Target directory+filename for the output image.  
[--model_path]                     # Path to the saved models.
[--num_matrices, default=256]      # Number of transformation matrices to be blended.
[--num_mlp, default=1]             # Number of MLPs to be used. For the latest version of our technique, keep it 1.
[--patch_size, default=[3,3]]      # Size of image patches to be processed.
[--laplacian_level, default=5]     # Number of Laplacian levels for frequency decomposition.
```


### Train:

```
python main.py --input_path=../Before.png --output_path=../After.png
```

### Test:

```
python test.py --test_path=../Input.png  --test_output_path=../Output.png
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

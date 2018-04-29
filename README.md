# paclitaxel-classify
Classificator of cells under influence of Paclitaxel (Taxol) using CNN.
Input: a photo taken from electron microscope.
Output: probability of beloning to each of the following classes accroding to concentration of Paclitaxel:
1. Control group
2. 0.1 mkg
3. 1 mkg
4. 5 mkg

Second semester scientific project in Saint Petersburg Academic University.

## Pipeline
1. Image segmentation (watershed)
2. Removing of bad artifacts (simple heuristics)
3. Data augmentation
    - Rotation
    - Flipping
    - Shifting
    - Scaling
 4. Learning (5 layers CNN in Keras)
 
## Architecture
input(78x78) -> conv(32x5x5) -> conv(32x5x5) -> maxpool(2x2) -> ReLU(1024) -> Softmax(4)

* Optimiser: SGD with Nesterov momentum
* Activation function: ReLU
* Loss function: Cross entropy

## Slides
* [English][1]
* [Russian][2]

[1]: https://goo.gl/BRD68A
[2]: https://goo.gl/AP4IoB

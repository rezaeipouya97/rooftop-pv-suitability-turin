# 01 — Point-cloud semantic segmentation

Classifies the photogrammetric point cloud of the Turin historic centre into three
classes (ground, building, tree) using a sparse-convolution deep-learning model,
then cleans the result and extracts building footprints.

## Pipeline order

1. **`generate_dtm.py`** — build a 1 m DTM from the raw cloud with the CSF (Cloth
   Simulation Filter) method. The DTM is used to compute each point's
   height-above-ground (HAG), one of the model's input features.
2. **`split_las.py`** — split a LAS tile into two halves (used to split the park
   tile into a training half and a validation half, since tree data was scarce).
3. **`dataset.py`** — lazy LAS dataset for training: random 20 m crops, 0.20 m
   voxelisation, 10 features per point (xyz + RGB + normals + HAG), class
   weighting and minority rejection for the rare tree class.
4. **`model.py`** — the TorchSparse UNet (32→64→128→256, ~5.8M parameters,
   10-channel input, 3-class output).
5. **`train.py`** — training loop (80 epochs, AMP, gradient clipping, best model
   chosen on validation mIoU).
6. **`inference_big_v2.py`** — runs the trained model on the full ~654M-point
   cloud: reads once into memmaps, tiles the area, slides overlapping crops, and
   majority-votes per point.
7. **`evaluate.py`** — confusion matrix + per-class metrics (Precision, Recall,
   F1, IoU) and overall accuracy / mIoU on the validation tiles.
8. **`extract_footprints.py`** — rasterises building points, applies morphology,
   connected-component labelling, and vectorises building footprint polygons.

## Result

On the balanced validation tile: **overall accuracy ≈ 98.5%, mIoU ≈ 95.9%**.

## Requirements

```
torch  torchsparse  laspy  rasterio  numpy  scipy  shapely  fiona  CSF
```

> Note: scripts expect input LAS/GeoTIFF data that is not included in this
> repository (see the main README).

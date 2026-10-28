<div align="center">
  <h1>Baby Cry Audio Classification</h1>
  <p>A notebook exploring classification of infant cry recordings with a CNN and a Wav2Vec2-based comparison model.</p>
  <p>
    <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square" alt="TensorFlow and Keras">
    <img src="https://img.shields.io/badge/PyTorch-Wav2Vec2-EE4C2C?style=flat-square" alt="PyTorch Wav2Vec2">
    <img src="https://img.shields.io/badge/librosa-Audio-8052A1?style=flat-square" alt="librosa audio">
  </p>
  <p><a href="#approach">Approach</a> · <a href="#open-the-notebook">Notebook</a> · <a href="#inspect-a-wav-folder">Data helper</a> · <a href="#results-and-limitations">Results and limitations</a></p>
</div>

## Approach

[`DE_G15_Code.ipynb`](DE_G15_Code.ipynb) contains two experiments:

1. **Deep CNN:** load class-labelled WAV files, generate augmented audio, extract normalized log-mel spectrograms with librosa, and train a TensorFlow/Keras classifier.
2. **Wav2Vec2 comparison:** load a pretrained Wav2Vec2 model with PyTorch and Transformers, add a classification head, and train with focal loss. The notebook references an external pretrained model and processor.

The CNN labels are `belly pain`, `burping`, `discomfort`, `hungry`, and `tired`. The notebook includes confusion matrices, classification reports and a comparison of its historical runs.

## Open the notebook

Open [`DE_G15_Code.ipynb`](DE_G15_Code.ipynb) in Jupyter or Google Colab. The original paths and Drive-mount cell target Colab. To reproduce an experiment, provide authorized WAV data, adjust `DATA_PATH` and/or `DATA_DIR` to its location, and review the cells before running them. The notebook expects class-named folders; the Wav2Vec2 section also supports `train/`, `val/`, and `test/` folders. Its installation cell lists dependencies for that section, while the CNN section also imports TensorFlow, librosa, scikit-learn, matplotlib, seaborn and joblib. There is no pinned environment for the full notebook, and this workflow has not been rerun for this README. The Wav2Vec2 section requires access to the referenced pretrained assets.

## Inspect a WAV folder

[`baby_cry_data.py`](baby_cry_data.py) extracts the Wav2Vec2 notebook's `load_full_data` and `load_data` functions into a small standard-library helper. It lists file paths and numeric labels; it does **not** open audio, train a model, run inference or fix the evaluation issues below. The notebook remains self-contained and does not import this helper.

For a folder with one subdirectory per class, such as `dataset/hungry/*.wav` and `dataset/tired/*.wav`:

```python
from baby_cry_data import load_full_data

paths, labels, classes = load_full_data("path/to/dataset")
print(classes, len(paths))
```

For a folder containing `train/`, `val/` and `test/`, each with class subdirectories:

```python
from baby_cry_data import load_data

train, validation, test, classes = load_data("path/to/splits")
print(classes, len(train[0]), len(validation[0]), len(test[0]))
```

Each split is a `(paths, numeric_labels)` pair. As in the notebook, only entries ending in lowercase `.wav` are selected, their order follows the filesystem, missing split directories yield empty lists, and **each split independently sorts and numbers its class directories**. A class missing from one split can therefore give the same number a different meaning. Check the actual folder layout before using these lists; this helper does not verify file type or content, consent, or split independence.

Run the helper tests from the repository root with Python 3; they create only empty synthetic placeholder files and use the standard library:

```bash
python -m unittest discover -s tests -v
```

## Results and limitations

The notebook's own comparison reports approximately **0.51 accuracy / 0.51 macro-F1** for its main model and **0.34 accuracy / 0.25 macro-F1** for the comparison model. These are historical notebook claims, not independently reproduced results.

The CNN augments each recording **before** the random train/validation/test split, so variants of one recording can appear in different splits and inflate evaluation. The provenance, consent, dataset availability and independence of recordings or infants across splits are not established in this repository. The pretrained model's training-data overlap is also unknown. In the pre-split Wav2Vec2 loader, each folder maps its own sorted class directories to numbers; missing a class folder can change the mapping across splits. Assess these issues and use group-separated evaluation before drawing conclusions about generalization.

This is an experimental classifier, not a medical or childcare diagnostic tool. The repository does not include the audio dataset or a packaged inference application.

# NoReFry

**No**n-linear **Re**sponse **F**unction Recove**ry**.

## Origin

This algorithm is used for calculation of material nonlinear response function.
It is applicable wherever the response function monotonically but non-linearly
increases with the deposited dose (e.g. fluorescence signal acquired from an
exposed LiF crystal, luminescence signal from a saturating Ce:YAG screen etc.).

Algorithm was originally developed for characterization of MHz X-ray beam via
method of desorption imprints. See the
[paper](https://doi.org/10.1364/OE.396755) to read more details. If you would
like to read just a summary, check the [poster](https://github.com/vojtavozda/NoReFry/blob/main/files/poster.pdf).

## Description

As the core of the algorithm is quite simple and most of the work consists of
data preparation which is individual for each case, no module has been
developed. Instead, description of an example is provided in
[`example.py`](https://github.com/vojtavozda/NoReFry/blob/main/example.py) and
Jupyter notebook
[`example.ipynb`](https://github.com/vojtavozda/NoReFry/blob/main/example.ipynb).
If it does not work here on GitHub check it at
[nbviewer.jupyter.org](https://nbviewer.jupyter.org/github/vojtavozda/NoReFry/blob/main/example.ipynb).
This example uses real data from the `data` folder used in the original
publication.

Please commit here if you find any mistakes or improvements.

## Citation

Feel free to use, modify and commit but do not forget to mention the authors in
your publications:

> V. Vozda, et al. "Characterization of megahertz X-ray laser beams by multishot desorption imprints in PMMA," *Opt. Express* **28**, 25664-25681 (2020)


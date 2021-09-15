### create conda environment `jlab_segmentation`
```
git clone https://github.com/AllenInstitute/ophys_etl_pipelines
cd ophys_etl_pipelines
git checkout staging/segmentation_dev
conda env create -f notebooks/segmentation/environment.yml
```

### for developers
This environment file installs the branch `staging/segmentation_dev`. This impacts any of the import statements in the notebook that begin with `from ophys_etl ...`. If one intends changes from the local namespace of `ophys_etl` to propogate into the notebook, then
```
pip install -e .
```
inside of this conda env will update the installed copy of `ophys_etl_pipelines` to the editable local copy. Restarting the kernel should make local changes visible.

### launch JupyterLab
```
conda activate jlab_segmentation
cd notebooks/segmentation
jupyter-lab
```
Open the notebook `segmentation_inspection.ipynb`
This notebook relies on the file `support_inspection_nb.py` in the correct relative path.

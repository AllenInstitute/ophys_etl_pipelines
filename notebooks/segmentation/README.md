### create env
```
git clone https://github.com/AllenInstitute/ophys_etl_pipelines
cd ophys_etl_pipelines
git checkout notebook_update
conda env create -f notebooks/segmentation/environment.yml
```

### launch as web app
```
cd notebooks/segmentation
voila segmentation_inspection.ipynb
```

### launch as normal jupyter notebook
```
cd notebooks/segmentation
jupyter-lab
```

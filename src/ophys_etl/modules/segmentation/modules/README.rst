Segmentation Steps
==================
- Detect (feature_vector_segmentation.py)
- Pre-merge filter (TBD)
- Merge (roi_merging.py)
- Filter (TBD)

Segmentation QC hdf5 output description
=======================================

group: "seed"
*************
the seeder works in conjunction with the detection algorithm. This group logs what seeds were provided, which were excluded, and the reasons for exclusion.

- provided_seeds: (row, col) coordinates of seeds served to the segmentation algorithm
- excluded_seeds: (row, col) coordinates of seeds not served to the segmentation algorithm
- exclusion_reason: (row, col) (str) reason the seed was not served.
- seed_image: (2D array) the image used for generating the seeds.
- attribute: (str) TBD - does not actually exist yet.

group: "detect"
***************
the detection by correlation and clustering stage.

- metric_image: (always same as seeder seed_image?)
- attribute: (str) the name of the attribute shown in the metric_image
- rois: (TBD) how to implement list of ROIs in hdf5

group: "merge"
**************
potential merging of adjacent ROIs

- rois: (TBD) how to implement list of ROIs in hdf5

group: "filter"
***************
post-process filters like size and ...

- rois: (TBD) how to implement list of ROIs in hdf5

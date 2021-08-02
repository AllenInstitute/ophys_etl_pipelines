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

- group_creation_time: (str) a timestamp for when this group was created.
- provided_seeds: (row, col) coordinates of seeds served to the segmentation algorithm
- excluded_seeds: (row, col) coordinates of seeds not served to the segmentation algorithm
- exclusion_reason: (row, col) (str) reason the seed was not served.
- seed_image: (2D array) the image used for generating the seeds.
- attribute: (str) TBD - does not actually exist yet.

group: "detect"
***************
the detection by correlation and clustering stage.

- group_creation_time: (str) a timestamp for when this group was created.
- metric_image: (always same as seeder seed_image?)
- attribute: (str) the name of the attribute shown in the metric_image
- rois: (str) utf-8 encoded serialized list of ExtractROI entries

group: "merge"
**************
potential merging of adjacent ROIs

- group_creation_time: (str) a timestamp for when this group was created.
- merger_ids: N x 2 array of (int) ROI IDs where each row is (dst, src) where
  src was merged into dst. the src_ID disappears from the rois and dst_ID is retained.
- rois: (str) utf-8 encoded serialized list of ExtractROI entries

group: "filter"
***************
post-process filters like size and ...

- group_creation_time: (str) a timestamp for when this group was created.
- rois: (str) utf-8 encoded serialized list of ExtractROI entries

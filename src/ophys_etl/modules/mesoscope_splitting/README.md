z_stack data description
===============================

This comment block contains what I have learned about the data model underlying those files, mostly by poking around in examples and seeing how they move through the code.

These files represent repeated scans of planes centered on planes of interest in the brain. There should be two planes of interest per file. The actual scans in these files are at different depths, presumably so that we can average data in three dimensions about the planes of interest.

The TIFF file itself contains a metadata field SI.hStackManager.zs which is a list of lists. Each sub-list contains two elements. The first element represents a z depth associated with the first plane. The second element represents a z depth associated with the second plane. For instance 

`SI.hStackManager.zs = [[9.9, 15.9], [10.0, 16.0], [10.1, 16.1]]`

would represent a local_stack focused on planes at 10 and 16 microns. Note that the `RoiGroups.imagingRoiGroup.rois.zs` metadata does not seem to matter in the cases of these files.

Each of these files corresponds to an ROI, where, in this case, ROI refers to an anatomical region of the brain (e.g. VISp) that was scanned. The metadata for the ROIs is encoded in

`metadata[1]['RoiGroups']['imagingRoiGroup']['rois']`

where metadata is the result of running

`tifffile.read_scanimage_metadata(open(local_zstack_filename, 'rb'))`

The list returned by `RoiGroups.imagingRoiGroup.rois` contains all of the ROIs for the mesoscope session. The ROI that this local_zstack file actually corresponds to is flagged with the `RoiGroups.imagingRoiGroup.rois.discretePlaneMode` field. Curiously, the ROI corresponding to this zstack file is the one with `discretePlaneMode = False` (presumably because `discretePlaneMode = True` means "try to match this plane based on z depth", which is not what we want to do in the volume scanning, since each ROI will have multiple scanned z values).

The TIFF splitting code actually associates experiments with anatomical ROIs using the `experiment['roi_index']` field, which is an integer corresponding to the ROI in `RoiGroups.imagingRoiGroup.rois` that the experiment is actually focused on. Put another way, in order for a local_zstack file to split properly `experiment['local_z_stack']` must point to a file in which `RoiGroups.imagingRoiGroup.roi[experiment['roi_index']].discretePlaneMode` is `False`
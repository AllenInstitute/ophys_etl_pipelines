ScanImage TIFF data model
===============================

Most of what is written here is based on experience. There is scant
official documentation to base this on but, from our experience, what
I am describing below is how TIFF files look coming off of the mesoscope rig
and more or less how to de-interleave them.

The important concepts to keep in mind are ROIs and z-values.

ROIs, in this case, refer to anatomical regions in the mouse brain
(not the ROIs of 2-photon segmentation fame). z-values refer to the
depths within the brain that are scanned in a given ROI. The ROIs and
z-values associated with a TIFF can be accessed through its metadata
which, generally, can be accessed via

```
import tifffile
metadata = tifffile.read_scanimage_metadata(open(path_to_tiff, 'rb'))
```

The resulting metadata structure will be a list with two elements in it.
`metadata[0]` is a dict of key-value pairs corresponding to the
configurations of ScanImage. `metadata[1]` is a dict that records the
structure of the ROIs scanned in this TIFF.

`metadata[0]['SI.hStackManager.zsAllActuators']` will be a list of
lists (or, in some edge cases, a list) of z-values in the order that they
were scanned by ScanImage. For instance, if there were 3 ROIs scanned at
z values (33, 22), (44, 11), and (88, 12) respectively, then
`metadata[0]['SI.hStackManager.zsAllActuators']` will be

```
[[33, 22], [44, 11], [88, 12]]
```

The ROIs for this TIFF can be found in
`metadata[1]['RoiGroups']['imagingRoiGroup']['rois']`. This will give you
a list of dicts, each of which represents an ROI scanned by this TIFF.
The important elements for a given ROI (from our perspective) are

`roi['zs']` a **sorted** list of zs scanned for this ROI (the sorting is
important; z values will appear in ascending order regardless of the order
in which ScanImage recorded them. To get the order in which a series of planes
were recorded, you must flatten
`metadata[0]['SI.hStackManager.zsAllActuators']`).

`roi['discretePlaneMode']` this will usuall be set to 1. For a z-stack TIFF
it will be set to 0 for the ROI to which the z-stack corresponds. This is
because `discretePlanMode==1` corresponds to an ROI that is only scanned at
discrete planes, i.e. at the planes defined by the user.
`discretePlaneMode==0` means the ROI is defined over an entire volume.
Data is collected whenever the planes specified by
`SI.hStackManager.zsAllActuators` crosses that volume (the ROI marked with
`discretePlaneMode==0`).

`roi['scanfields'][ii]['centerXY']` 'scanfields' gives you a list of all of
the planes scanned in the ROI. 'centerXY' gives you the 2-dimensional center
of those planes. That may be the best way to associate an ROI in one file
with an ROI in another, given that, for instance, surface TIFFs have
z-values that are vastly unlike the z-values in depth, timeseries, and
z-stack tiffs since surface TIFFs are set to scan the surface of the brain.

Data model
===============================
There are four main types of TIFF file that we need to process. I will
describe them below. These descriptions are based
[documentation](http://confluence.corp.alleninstitute.org/pages/viewpage.action?spaceKey=IT&title=MesoScope+and+DeepScope+Session+Creation+and+Handoff&preview=/46568913/93455697/20201023_Mesoscope_experiment_and%20_data_splitting%20specifications.docx)
provided by Natalia Orlova. **Note:** the linked document implies that if a
z-value is specified more than once in a TIFF file, images from that plane
are all stored on a single page of the TIFF file with a gap of dead pixels
between them. Natalia and I have inspected many examples of TIFFs coming
off of the rig and we agree that this is no longer happening, probably
due to a version change in ScanImage. Data in the TIFF file is actually
stored one image per page. The pages are recorded in the order of the
flattened `SI.hStackManager.zsAllActuators` array.

### _averaged_surface.tiff ###

A TIFF containing a stack of images taken at the surface of the ROI. There
should only be one set of images per ROI, no matter how many z-planes
are sampled per ROI in the other TIFF files (see metadata examples below).
When processing these files, we need to take all of the frames for a given
ROI and average them together into a single summary image.

### _averaged_depth.tiff ###

A TIFF containing, for each scanned z-value, a stack of images at that z-value.
There should be images for each (ROI, z-value) pair. Our job when processing
these is to take every page corresponding to each plane and average them
together so that there is a summary image for each (ROI, z-value) pair.

### _local_z_stack.tiff ###

Quoting Natalia: "A local z-stack, acquired for each plane in each ROI. This
data contains images from the +/- 30 microns around the plane of interest...
[There will be] one TIFF file per imaging group [an imaging group is pair
of z-values associated in `SI.hStackManager.zsAllActuators`]"

Our job when processing these files is to de-interleave them and write all of
the frames for a given (ROI, z-value) pair to an HDF5 file, much like the
timeseries data that becomes the video.

Because each _local_z_stack.tiff file corresponds to only two z-values, and
not necessarily to a single ROI (ROIs can be sampled at more than two z-values),
you need to determine which ROI the file corresponds to by looking at the
list of ROIs in the metadata and finding the one with `discretePlaneMode==0`
(see metadata examples below).

### _timeseries.tiff ###

This file contains the timeseries data for all ROIs at all z-values.
Our job is to de-interleave them and write the video data for each
(ROI, z-value) pair to a separate HDF5 file.


Metadata examples
===============================

### 4x2 session ###

Here we examine a 4x2 (4 ROIs x 2 z-values) session.

This is the depth TIFF

```
>>> import tifffile
>>> fname='1071644869_averaged_depth.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname,'rb'))
>>>
>>> m[0]['SI.hStackManager.zsAllActuators']
[[204, 84], [304, 184], [264, 144], [274, 159]]
```

With the corresponding ROIs:
```
>>> rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
>>> type(rois)
<class 'list'>
>>> len(rois)
4
>>> for r in rois:
...     print(r['zs'])
...
[84, 204]
[184, 304]
[144, 264]
[159, 274]
>>>
```
Note that the z-values listed with the ROIs are all sorted.
The order of the z-values in `SI.hStackManager.zsAllActuators`
represents the order in which the planes were actually scanned by
the rig.

Here is the surface TIFF associated with that session

```
>>> fname = '1071644869_averaged_surface.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[[29, 0], [129, 0], [89, 0], [99, 0]]
>>>
```
**Note:** those zeros are all placeholders.


A z-stack TIFF associated with that session (this one contains the z-stacks
for the 0th ROI with z-values at 84 and 204 microns).
```
>>> fname = '1071644869_local_z_stack0.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[[174, 54], [174.75, 54.75], [175.5, 55.5], [176.25, 56.25], [177, 57], [177.75, 57.75], [178.5, 58.5], [179.25, 59.25], [180, 60], [180.75, 60.75], [181.5, 61.5], [182.25, 62.25], [183, 63], [183.75, 63.75], [184.5, 64.5], [185.25, 65.25], [186, 66], [186.75, 66.75], [187.5, 67.5], [188.25, 68.25], [189, 69], [189.75, 69.75], [190.5, 70.5], [191.25, 71.25], [192, 72], [192.75, 72.75], [193.5, 73.5], [194.25, 74.25], [195, 75], [195.75, 75.75], [196.5, 76.5], [197.25, 77.25], [198, 78], [198.75, 78.75], [199.5, 79.5], [200.25, 80.25], [201, 81], [201.75, 81.75], [202.5, 82.5], [203.25, 83.25], [204, 84], [204.75, 84.75], [205.5, 85.5], [206.25, 86.25], [207, 87], [207.75, 87.75], [208.5, 88.5], [209.25, 89.25], [210, 90], [210.75, 90.75], [211.5, 91.5], [212.25, 92.25], [213, 93], [213.75, 93.75], [214.5, 94.5], [215.25, 95.25], [216, 96], [216.75, 96.75], [217.5, 97.5], [218.25, 98.25], [219, 99], [219.75, 99.75], [220.5, 100.5], [221.25, 101.25], [222, 102], [222.75, 102.75], [223.5, 103.5], [224.25, 104.25], [225, 105], [225.75, 105.75], [226.5, 106.5], [227.25, 107.25], [228, 108], [228.75, 108.75], [229.5, 109.5], [230.25, 110.25], [231, 111], [231.75, 111.75], [232.5, 112.5], [233.25, 113.25], [234, 114]]
>>>
```

and the associated ROIs
```
>>> rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
>>> for r in rois:
...     print(r['zs'], r['discretePlaneMode'])
...
29 0
129 1
89 1
99 1
>>>
```
Note that the ROI associated with this z-stack is marked with
`discretePlaneMode == 0`. Also note that the `zs` associated with the ROIs
are somewhat meaningless here. In order to figure out which z-values
the z-stack was scanning, you need to look at the de-interleaved
`SI.hStackManager.zsAllActuators`.

If we examine all of the z_stack TIFFs associated with each session, we
see that there is one z_stack file associated with each ROI
```
>>> for stack_path in z_stack_list:
...     m = tifffile.read_scanimage_metadata(open(stack_path, 'rb'))
...     print(stack_path.name)
...     rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
...     for r in rois:
...         print('    ',r['zs'],r['discretePlaneMode'])
...
1071644869_local_z_stack0.tiff
     29 0
     129 1
     89 1
     99 1
1071644869_local_z_stack1.tiff
     29 1
     129 0
     89 1
     99 1
1071644869_local_z_stack2.tiff
     29 1
     129 1
     89 0
     99 1
1071644869_local_z_stack3.tiff
     29 1
     129 1
     89 1
     99 0
>>>
```

### 2x4 session ###

Here we examine a 2x4 (2 ROIs x 4 z-values) session.


Here is the depth TIFF
```
>>> fname='1161151478_averaged_depth.tiff'
>>> import tifffile
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[[230, -11], [170, 69], [290, -11], [190, 89]]
>>>
>>> rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
>>> type(rois)
<class 'list'>
>>> len(rois)
2
>>> for r in rois:
...     print(r['zs'])
...
[-11, 69, 170, 230]
[-11, 89, 190, 290]
>>>
```
Note that, in this case, the z_values in `SI.hStackManager.zsAllActuators`
are ordered like
```
[[roi0_z0, roi0_z1], [roi0_z2, roi0_z3], [roi_1_z0, roi1_z1], ...]
```
and that the `zs` in the ROI dict are sorted.

Here is the surface TIFF
```
>>> fname='1161151478_averaged_surface.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[[-105, 0], [-85, 0]]
>>>
```

Here is one of the z_stack TIFFs
```
>>> fname='1161151478_local_z_stack0.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[[200, -41], [200.75, -40.25], [201.5, -39.5], [202.25, -38.75], [203, -38], [203.75, -37.25], [204.5, -36.5], [205.25, -35.75], [206, -35], [206.75, -34.25], [207.5, -33.5], [208.25, -32.75], [209, -32], [209.75, -31.25], [210.5, -30.5], [211.25, -29.75], [212, -29], [212.75, -28.25], [213.5, -27.5], [214.25, -26.75], [215, -26], [215.75, -25.25], [216.5, -24.5], [217.25, -23.75], [218, -23], [218.75, -22.25], [219.5, -21.5], [220.25, -20.75], [221, -20], [221.75, -19.25], [222.5, -18.5], [223.25, -17.75], [224, -17], [224.75, -16.25], [225.5, -15.5], [226.25, -14.75], [227, -14], [227.75, -13.25], [228.5, -12.5], [229.25, -11.75], [230, -11], [230.75, -10.25], [231.5, -9.5], [232.25, -8.75], [233, -8], [233.75, -7.25], [234.5, -6.5], [235.25, -5.75], [236, -5], [236.75, -4.25], [237.5, -3.5], [238.25, -2.75], [239, -2], [239.75, -1.25], [240.5, -0.5], [241.25, 0.25], [242, 1], [242.75, 1.75], [243.5, 2.5], [244.25, 3.25], [245, 4], [245.75, 4.75], [246.5, 5.5], [247.25, 6.25], [248, 7], [248.75, 7.75], [249.5, 8.5], [250.25, 9.25], [251, 10], [251.75, 10.75], [252.5, 11.5], [253.25, 12.25], [254, 13], [254.75, 13.75], [255.5, 14.5], [256.25, 15.25], [257, 16], [257.75, 16.75], [258.5, 17.5], [259.25, 18.25], [260, 19]]
>>>
>>> rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
>>> for r in rois:
...     print(r['zs'], r['discretePlaneMode'])
...
-105 0
-85 1
>>>
```

In this case, we see that there are two z_stack TIFF files associated
with each ROI (since each ROI is sampled at more than two z-values).
In order to determine which z-values are associated with which files,
we must examine the means of the de-interleaved
`SI.hstackManager.zsAllActuators`
```
>>> for stack_path in z_stack_list:
...     m = tifffile.read_scanimage_metadata(open(stack_path, 'rb'))
...     print(stack_path.name)
...     rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
...     for r in rois:
...         print('    ',r['zs'],r['discretePlaneMode'])
...
1161151478_local_z_stack0.tiff
     -105 0
     -85 1
1161151478_local_z_stack1.tiff
     -105 0
     -85 1
1161151478_local_z_stack2.tiff
     -105 1
     -85 0
1161151478_local_z_stack3.tiff
     -105 1
     -85 0
>>>
```

### 1x6 session ###

Here is the same metadata for a 1x6 (1 ROI x 6 z-values) session.
```
>>> fname = '1158510243_averaged_depth.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[[310, 67], [260, 117], [210, 167]]
>>> rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
>>> type(rois)
<class 'dict'>
>>> rois['zs']
[67, 117, 167, 210, 260, 310]
>>>
```
Note that, in this case, `rois` is not a list; it is just a dict
because there is only one ROI.

Here is the surface TIFF

```
>>> fname='1158510243_averaged_surface.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> m[0]['SI.hStackManager.zsAllActuators']
[10, 0]
>>>
```
Note that `SI.hStackManager.zsAllActuators` is not a list of lists,
it is just a list (again: because there is only one ROI).

Here is one of the associated z-stack TIFFs
```
>>> fname = '1158510243_local_z_stack1.tiff'
>>> m = tifffile.read_scanimage_metadata(open(fname, 'rb'))
>>> rois = m[1]['RoiGroups']['imagingRoiGroup']['rois']
>>> type(rois)
<class 'dict'>
>>> rois['zs']
10
>>> rois['discretePlaneMode']
0
>>>
```

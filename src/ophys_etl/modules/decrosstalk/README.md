Running the decrosstalk pipeline will automatically result in some quality
control artifacts being produced in the `qc_output_dir` specified in the
input.json file.

# QC data file

For each plane processed, an HDF5 file named like

```
{ophys_experiment_id}_qc_data.h5
```

will be produced containing summary QC data for each ROI in the plane.
The schema of that file is as follows:

- `data['paired_plane']` -- the ophys_experiment_id of the plane coupled to
this one

- `data['ROI']` -- the group containing all of the data pertaining to ROIs
in this plane

- `data['ROI/{roi_id}']` -- the group containing all of the data pertaining
to a specific ROI

- `data['ROI/{roi_id}/is_ghost']` -- a boolean denoting if the ROI is a ghost

- `data['ROI/{roi_id}/valid_raw_trace']` -- a boolean denoting if the ROI
had a valid raw trace

- `data['ROI/{roi_id}/valid_raw_active_trace']` -- a boolean denoting if the
measure of activity in the ROI's raw trace gave a valid result

- `data['ROI/{roi_id}/valid_unmixed_trace']` -- a boolean denoting if the
ROI had a valid unmixed trace (after decrosstalking)

- `data['ROI/{roi_id}/valid_unmixed_active_trace']` -- a boolean denoting
if the measure of activity in the unmixed trace gave a valid result

- `data['ROI/{roi_id}/roi']` -- the group containing all of the data measured
to ROI specified by {roi_id}

- `data['ROI/{roi_id}/roi/raw/signal/trace']` -- the raw signal trace
associated with the ROI

- `data['ROI_ID/{roi_id}/roi/raw/signal/events']` -- the indices of the active
events in the raw signal trace

- `data['ROI_ID/{roi_id}/roi/raw/crosstalk/trace']` -- the raw crosstalk
trace associated with the ROI

- `data['ROI_ID/{roi_id}/roi/raw/crosstalk/events']` -- the indices of active
events in the raw crosstalk trace

- `data['ROI/{roi_id}/roi/unmixed/converged']` -- a boolean indicating
whether or not ICA converged for this ROI

- `data['ROI/{roi_id}/roi/unmixed/mixing_matrix']` -- the mixing matrix used
to unmix this ROI

- `data['ROI/{roi_id}/roi/unmixed/signal/trace']` -- the trace for the
unmixed signal

- `data['ROI/{roi_id}/roi/unmixed/signal/events']` -- the indices of active
events in the unmixed signal

- `data['ROI/{roi_id}/roi/unmixed/cosstalk/trace']` -- the trace of the unmixed
crosstalk

- `data['ROI/{roi_id}/roi/unmixed/crosstalk/events']` -- the indices of
active events in the unmixed crosstalk

- `data['ROI/{roi_id}/roi/unmixed/unclipped_signal/trace']` -- the unclipped
signal trace ("unclipped" meaning "not corrected for negative spikes")

- `data['ROI/{roi_id}/roi/unmixed/poorly_converged_signal']` -- the poorly
converged signal trace (where 'converged' is False)

- `data['ROI/{roi_id}/roi/unmixed/poorly_converged_crosstalk']` -- the
poorly-converged crosstalk trace (where 'converged' is False)

- `data['ROI/{roi_id}/roi/unmixed/poorly_converged_mixing_matrix']` -- the
poorly-converged mixing matrix (where 'converged' is False)

- `data['ROI/{roi_id}/neuropil']` -- the group containing all of the data
measured from neuropil around ROI {roi_id} (mostly a mirror of
data['ROI/{roi_id}/roi'], just lacking `events` entries, since activity
detection was not run on neuropil traces)

# QC figures

In addition to the HDF5 file, QC figures will be automatically generated in
the `qc_output_dir`

A figure named like
```
{ophys_session_id}_roi_fig.png
```
will be generated, showing maximum projection figures of each plane in the
session with ROIs overlaid.

For each pair of ROIs that overlap in pixel space, there will also be
detailed comparison figures named like
```
{roi_id_0}_{roi_id_1}_comparison.png
```
These will be written to sub directories named like
```
{qc_output_dir}/{ophys_experiment_id_0}_{ophys_experiment_id_1}_roi_pairs/
```
where the two `ophys_experiment_id` values refer to the planes in which the
ROIs wer identified.

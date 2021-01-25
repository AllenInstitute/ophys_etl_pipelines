import os
import h5py
import numpy as np
import argschema
from ophys_etl.decrosstalk.ophys_plane import DecrosstalkingOphysPlane
import ophys_etl.decrosstalk.decrosstalk_schema as decrosstalk_schema
from ophys_etl.decrosstalk.decrosstalk import run_decrosstalk


class DecrosstalkWrapper(argschema.ArgSchemaParser):

    default_schema = decrosstalk_schema.DecrosstalkInputSchema
    default_output_schema = decrosstalk_schema.DecrosstalkOutputSchema

    def run(self):
        clobber = True

        cache_dir = self.args['qc_output_dir']
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.isdir(cache_dir):
            raise RuntimeError("\n%s\nis not a dir" % cache_dir)

        final_output = {}
        final_output['ophys_session_id'] = self.args['ophys_session_id']

        coupled_planes = []

        for meta_pair in self.args['coupled_planes']:
            output_pair = {}

            imaging_id = meta_pair['ophys_imaging_plane_group_id']
            output_pair['ophys_imaging_plane_group_id'] = imaging_id

            group_order = meta_pair['group_order']
            output_pair['group_order'] = group_order

            input_pair = meta_pair['planes']
            plane_A = DecrosstalkingOphysPlane.from_schema_dict(input_pair[0])
            plane_B = DecrosstalkingOphysPlane.from_schema_dict(input_pair[1])

            plane_pair = []
            plane_group_A = (plane_A, input_pair[0])
            plane_group_B = (plane_B, input_pair[1])

            for p_pair in [(plane_group_A, plane_group_B),
                           (plane_group_B, plane_group_A)]:
                plane_0 = p_pair[0][0]
                plane_1 = p_pair[1][0]
                output_schema = p_pair[0][1]

                p0 = {}
                p0['ophys_experiment_id'] = plane_0.experiment_id

                roi_fname = output_schema['output_roi_trace_file']
                p0['output_roi_trace_file'] = roi_fname

                neuropil_fname = output_schema['output_neuropil_trace_file']
                p0['output_neuropil_trace_file'] = neuropil_fname
                (flags_0,
                 traces_0) = run_decrosstalk(plane_0,
                                             plane_1,
                                             cache_dir=cache_dir,
                                             clobber=clobber)

                invalid_roi = set()
                for field in flags_0.keys():
                    p0[field] = flags_0[field]
                    for roi_id in flags_0[field]:
                        invalid_roi.add(roi_id)
                plane_pair.append(p0)

                for k in ('roi', 'neuropil'):
                    out_fname = output_schema['output_%s_trace_file' % k]
                    if k not in traces_0:
                        continue
                    data = []
                    roi_names = []
                    roi_list = list(traces_0[k].keys())
                    roi_list.sort()
                    for roi_id in roi_list:
                        if roi_id not in invalid_roi:
                            roi_names.append(roi_id)
                            data.append(traces_0[k][roi_id]['signal'])

                    # add in np.arrays of NaNs for the invalid ROIs
                    # so that downstream modules don't get confused
                    # when the number of ROIs in the experiment changes
                    # (we are adding NaNs because there is no well-defind
                    # 'unmixed' trace for these cases; the invalid ROI flags
                    # added to LIMS by this module will exempt these traces
                    # from further processing)
                    if len(data) > 0:
                        n_t = len(data[0])
                    else:
                        n_t = 10000
                    for roi_id in invalid_roi:
                        roi_names.append(roi_id)
                        data.append(np.NaN*np.ones(n_t, dtype=float))

                    roi_names = np.array(roi_names)
                    data = np.array(data)
                    with h5py.File(out_fname, 'w') as out_file:
                        out_file.create_dataset('roi_names', data=roi_names)
                        out_file.create_dataset('data', data=data)

            output_pair['planes'] = plane_pair
            coupled_planes.append(output_pair)

        final_output['coupled_planes'] = coupled_planes
        self.output(final_output, indent=2, sort_keys=True)


if __name__ == "__main__":
    wrapper = DecrosstalkWrapper()
    wrapper.run()

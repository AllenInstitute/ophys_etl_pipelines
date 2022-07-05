import h5py
import os
from pathlib import Path


if __name__ == "__main__":

    exp_ids = [858370800, 859125454,
               841511175, 842852916,
               842462049]

    base_dir = Path("/allen/aibs/informatics/chris.morrison/ticket-504"
                    "/trace_extraction_2022")
    for exp_id in exp_ids:
        # trace files
        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        trace_output_dir = base_dir / "traces_2022" / str(exp_id)
        if not trace_output_dir.exists():
            print(f'Creating Trace output dir {str(trace_output_dir)}')
            os.makedirs(trace_output_dir)
        output_roi_trace_file = (
            trace_output_dir / "roi_traces.h5")
        if not output_roi_trace_file.exists():
            with h5py.File(output_roi_trace_file, "w") as h5_file:
                h5_file.create_dataset(name="data", data=[], dtype="f8")
                h5_file.create_dataset(name="roi_names",
                                       data=[],
                                       dtype=ref_dtype)

        output_neuropil_trace_file = Path(
            trace_output_dir / "neuropil_traces.h5")
        if not output_neuropil_trace_file.exists():
            with h5py.File(output_neuropil_trace_file, "w") as h5_file:
                h5_file.create_dataset(name="data", data=[], dtype="f8")
                h5_file.create_dataset(name="roi_names",
                                       data=[],
                                       dtype=ref_dtype)

        # demix files
        demix_output_dir = base_dir / "demix_2022" / str(exp_id)
        if not demix_output_dir.exists():
            print(f'Creating demix output dir {str(demix_output_dir)}')
            os.makedirs(demix_output_dir)
        output_demix_file = (
            demix_output_dir / f"{exp_id}_demixed_traces.h5")
        if not output_demix_file.exists():
            with h5py.File(output_demix_file, "w") as h5_file:
                h5_file.create_dataset(name="data", data=[], dtype="f8")
                h5_file.create_dataset(name="roi_names",
                                       data=[],
                                       dtype="S4")

        # neuropil correct files
        neuropil_output_dir = base_dir / "neuropil_2022" / str(exp_id)
        if not neuropil_output_dir.exists():
            print(f'Creating neuropil output dir {str(neuropil_output_dir)}')
            os.makedirs(neuropil_output_dir)
        output_neuropil_file = (
            neuropil_output_dir / "neuropil_correction.h5")
        if not output_neuropil_file.exists():
            with h5py.File(output_neuropil_file, "w") as h5_file:
                h5_file.create_dataset(name="FC", data=[], dtype="f8")
                h5_file.create_dataset(name="RMSE", data=[], dtype="f8")
                h5_file.creat_dataset(name="r", data=[], dtype="f8")
                h5_file.create_group(name="r")
                h5_file.create_dataset(name="roi_names",
                                       data=[],
                                       dtype="S4")

        # dff files
        dff_output_dir = base_dir / "dff_2022" / str(exp_id)
        if not dff_output_dir.exists():
            print(f'Creating neuropil output dir {str(dff_output_dir)}')
            os.makedirs(dff_output_dir)
        output_dff_file = (
            dff_output_dir / f"{exp_id}_dff.h5")
        if not output_dff_file.exists():
            with h5py.File(output_dff_file, "w") as h5_file:
                h5_file.create_dataset(name="data", data=[], dtype="f8")
                h5_file.create_dataset(name="num_small_baseline_frames",
                                       data=[],
                                       dtype="i8")
                h5_file.create_dataset(name="roi_names",
                                       data=[],
                                       dtype="S4")
                h5_file.create_dataset(name="sigma_dff",
                                       data=[],
                                       dtype="f8")

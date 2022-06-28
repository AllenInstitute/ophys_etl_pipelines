import json
import pandas as pd
from subprocess import Popen


if __name__ == "__main__":

    metadata = pd.read_csv("/allen/aibs/informatics/chris.morrison/ticket-504/"
                           "other_mice_meta.csv")
    segmentation_dir = "/allen/programs/mindscope/workgroups/surround/segmentation"
    for idx, row in metadata.iterrows():
        experiment_id = row["id"]
        frame_rate = row["movie_frame_rate_hz"]
        storage_dir = row["storage_directory"]
        output_str = (
            "#!/bin/bash\n"
            "#SBATCH --partition=braintv\n"
            "#SBATCH --nodes=1 --cpus-per-task=32 --mem=250G\n"
            "#SBATCH --time=10:00:00\n"
            "#SBATCH --export=NONE\n"
            f"#SBATCH --job-name=ssf_traces_{experiment_id}\n"
            "#SBATCH --output=/allen/aibs/informatics/chris.morrison/"
            f"ticket-504/{experiment_id}.log\n"
            "#SBATCH --mail-type=NONE\n"
            "#SBATCH --tmp=128G\n"
            "source /home/chris.morrison/.bashrc\n"
            "export TMPDIR=/scratch/fast/${SLURM_JOB_ID}\n"
            "conda activate allensdk\n"
            "python /allen/aibs/informatics/chris.morrison/src/"
            "ophys_etl_pipelines/src/ophys_etl/scripts/ssf_post_processing_3_more_mice.py "
            "--output_path=/allen/programs/mindscope/workgroups/surround/"
            "trace_extraction_2022/ "
            f"--experiment_id={experiment_id} "
            f"--movie_frame_rate_hz={frame_rate} "
            f"--motion_corrected_data_dir={storage_dir}/processed/ "
            f"--segmentation_data_dir={segmentation_dir}\n")
        with open(f"traces_{experiment_id}.slrum", "w") as text_file:
            text_file.write(output_str)
        job = Popen(f"sbatch traces_{experiment_id}.slrum", shell=True)
        job.wait()

import argschema
import json
from pathlib import Path

from ophys_etl.modules.create_slapp_inputs.schemas import (
        ExperimentOutputSchema, SlappTransformInputSchema,
        SlappTransformOutputSchema)
from ophys_etl.modules.create_slapp_inputs.utils import (
        select_rois, populate_experiments_rglob)


class SlappTransformInput(argschema.ArgSchemaParser):
    default_schema = SlappTransformInputSchema
    default_output_schema = SlappTransformOutputSchema

    def run(self):
        experiments = self.args['experiments']

        # this arg triggers a search for 2 input files
        if 'input_rootdir' in self.args:
            self.logger.info("finding paths for ROIs and traces in "
                             f"{self.args['input_rootdir']}")
            experiments = populate_experiments_rglob(
                    experiments,
                    rootdir=Path(self.args['input_rootdir']),
                    binarized_filename=self.args['binarized_filename'],
                    trace_filename=self.args['trace_filename'])

        # this arg triggers a listing of all included ROIs
        # and a random subselection of those
        if 'n_roi_total' in self.args:
            self.logger.info(f"randomly selecting {self.args['n_roi_total']} "
                             "ROIs from non-excluded ROIs")
            experiments = select_rois(experiments, self.args['n_roi_total'],
                                      self.args['random_seed'])

        # apply global label to each ROI and output
        global_counter = self.args['global_id_offset']
        outdir = Path(self.args['output_dir'])
        outjpaths = []
        for experiment in experiments:
            if 'local_ids' in experiment:
                local_to_global = {}
                for local in experiment['local_ids']:
                    local_to_global[local] = global_counter
                    global_counter += 1
                experiment['local_to_global_roi_id_map'] = local_to_global
                experiment.pop('local_ids')

                # validate
                ExperimentOutputSchema().load(experiment)

                # output data
                outjpaths.append(
                        str(outdir / f"{experiment['experiment_id']}.json"))
                with open(outjpaths[-1], "w") as f:
                    json.dump(experiment, f, indent=2)
                self.logger.info(f"wrote {outjpaths[-1]}")

        self.output({"outputs": outjpaths})
        self.logger.info(f"wrote {self.args['output_json']}")


if __name__ == "__main__":  # pragma: nocover
    sti = SlappTransformInput()
    sti.run()

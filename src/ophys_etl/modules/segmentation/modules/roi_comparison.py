import pathlib
import argschema
from marshmallow import post_load

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    create_roi_v_background_grid)


class ROIComparisonSchema(argschema.ArgSchema):

    background_paths = argschema.fields.List(
            argschema.fields.InputFile,
            cli_as_single_argument=True,
            required=True,
            description=("list of paths to pkl or png background images"))

    background_names = argschema.fields.List(
            argschema.fields.String,
            cli_as_single_argument=True,
            required=False,
            default=None,
            allow_none=True,
            description=("names of background images as they are to appear "
                         "in plot (if None, will use the basenames "
                         "from the roi_paths)"))

    roi_paths = argschema.fields.List(
            argschema.fields.InputFile,
            cli_as_single_argument=True,
            required=True,
            description=("list of paths to json files with ROIs to compare"))

    roi_names = argschema.fields.List(
            argschema.fields.String,
            cli_as_single_argument=True,
            required=False,
            default=None,
            allow_none=True,
            description=("names of ROI sets as they are to appear "
                         "in plot (if None, will use the basenames "
                         "from the roi_paths)"))

    plot_output = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Where to save the output plot"))

    attribute_name = argschema.fields.Str(
            required=False,
            default='filtered_hnc_Gaussian',
            description=("name of attribute to plot in a background image"))

    @post_load
    def verify_names_and_paths(self, data, **kwargs):

        if data['roi_names'] is None:
            data['roi_names'] = [str(pathlib.Path(pth).name)
                                 for pth in data['roi_paths']]

        if data['background_names'] is None:
            data['background_names'] = [str(pathlib.Path(pth).name)
                                        for pth in data['background_paths']]
        msg = ''
        is_valid = True
        if len(data['roi_names']) != len(data['roi_paths']):
            is_valid = False
            msg += f"{len(data['roi_names'])} roi_names, but "
            msg += f"{len(data['roi_paths'])} roi_paths; "
            msg += "should be the same length\n"
        if len(data['background_names']) != len(data['background_paths']):
            is_valid = False
            msg += f"{len(data['background_names'])} background_names, but "
            msg += f"{len(data['background_paths'])} background_paths; "
            msg += "should be the same length\n"
        if not is_valid:
            raise RuntimeError(msg)
        return data


class ROIComparisonEngine(argschema.ArgSchemaParser):

    default_schema = ROIComparisonSchema

    def run(self):

        # colors that showup reasonably well against our
        # grayscale background images
        color_list = [(0, 255, 0),
                      (51, 153, 55),
                      (51, 255, 255)]

        invalid_color = (235, 52, 52)

        bckgd_paths = [pathlib.Path(p)
                       for p in self.args['background_paths']]

        roi_paths = [pathlib.Path(p)
                     for p in self.args['roi_paths']]

        fig = create_roi_v_background_grid(
                    bckgd_paths,
                    self.args['background_names'],
                    roi_paths,
                    self.args['roi_names'],
                    color_list,
                    invalid_color=invalid_color,
                    attribute_name=self.args['attribute_name'])

        fig.savefig(self.args['plot_output'])


if __name__ == "__main__":
    comparison_engine = ROIComparisonEngine()
    comparison_engine.run()

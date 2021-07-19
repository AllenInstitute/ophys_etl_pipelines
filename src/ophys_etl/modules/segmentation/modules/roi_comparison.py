import pathlib
import argschema

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
            required=True,
            description=("names of background images as they are to appear "
                         "in plot"))

    roi_paths = argschema.fields.List(
            argschema.fields.InputFile,
            cli_as_single_argument=True,
            required=True,
            description=("list of paths to json files with ROIs to compare"))

    roi_names = argschema.fields.List(
            argschema.fields.String,
            cli_as_single_argument=True,
            required=True,
            description=("names of ROI sets as they are to appear "
                         "in plot"))

    plot_output = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Where to save the output plot"))

    attribute_name = argschema.fields.Str(
            required=False,
            default='filtered_hnc_Gaussian',
            description=("name of attribute to plot in a background image"))


class ROIComparisonEngine(argschema.ArgSchemaParser):

    default_schema = ROIComparisonSchema

    def run(self):

        # colors that showup reasonably well against our
        # grayscale background images
        color_list = [(0, 255, 0),
                      (255, 128, 0),
                      (51, 255, 255),
                      (255, 51, 255)]

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
                    attribute_name=self.args['attribute_name'])

        fig.savefig(self.args['plot_output'])


if __name__ == "__main__":
    comparison_engine = ROIComparisonEngine()
    comparison_engine.run()

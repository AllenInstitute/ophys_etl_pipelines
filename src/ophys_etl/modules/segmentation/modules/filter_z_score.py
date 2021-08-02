import pathlib

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.filter.schemas import (
    ZvsBackgroundSchema)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    FilterRunnerBase,
    ZvsBackgroundFilter)


class ZvsBackgroundFilterRunner(FilterRunnerBase):
    default_schema = ZvsBackgroundSchema

    def get_filter(self):
        metric_img = graph_to_img(
                        pathlib.Path(self.args['graph_input']),
                        attribute_name=self.args['attribute_name'])

        return ZvsBackgroundFilter(
                    metric_img,
                    self.args['min_z'],
                    self.args['n_background_factor'],
                    self.args['n_background_minimum'])


if __name__ == "__main__":
    runner = ZvsBackgroundFilterRunner()
    runner.run()

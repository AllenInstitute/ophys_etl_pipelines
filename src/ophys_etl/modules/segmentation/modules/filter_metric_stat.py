import pathlib

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.filter.schemas import (
    StatFilterSchema)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    FilterRunnerBase,
    ROIMetricStatFilter)


class StatFilterRunner(FilterRunnerBase):
    default_schema = StatFilterSchema

    def get_filter(self):
        metric_img = graph_to_img(
                        pathlib.Path(self.args['graph_input']),
                        attribute_name=self.args['attribute_name'])

        return ROIMetricStatFilter(
                   metric_img,
                   self.args['stat_name'],
                   self.args['min_value'],
                   self.args['max_value'])


if __name__ == "__main__":
    runner = StatFilterRunner()
    runner.run()

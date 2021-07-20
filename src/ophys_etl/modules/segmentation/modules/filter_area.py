from ophys_etl.modules.segmentation.filter.schemas import (
    AreaFilterSchema)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    FilterRunnerBase,
    ROIAreaFilter)


class AreaFilterRunner(FilterRunnerBase):
    default_schema = AreaFilterSchema

    def get_filter(self):
        return ROIAreaFilter(min_area=self.args['min_area'],
                             max_area=self.args['max_area'])


if __name__ == "__main__":

    runner = AreaFilterRunner()
    runner.run()

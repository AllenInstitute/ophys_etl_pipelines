from ophys_etl.workflows.pipeline_modules.roi_classification.create_train_test_split import ( # noqa E501
    CreateTrainTestSplitModule,
)


class TestCreateTrainTestSplitModule:

    def test_inputs(
        self,
        mock_thumbnails_dir
    ):
        """Test that inputs are correctly formatted
        for input into the module.
        """

        mod = CreateTrainTestSplitModule(
            docker_tag="main",
            thumbnails_dir={
                0: mock_thumbnails_dir,
                1: mock_thumbnails_dir
            },
            exp_roi_meta_map={
                0: {
                    0: {
                        'is_inside_motion_border': True
                    }
                },
                1: {
                    0: {
                        'is_inside_motion_border': False
                    },
                    1: {
                        'is_inside_motion_border': True
                    }
                }
            }
        )

        mod.inputs

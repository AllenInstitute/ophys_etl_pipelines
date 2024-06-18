import os
from pathlib import Path


def setup_app_config(
    ophys_workflow_app_config_path: Path,
    test_di_base_model_path: Path
):
    """
    Sets up app config for testing

    Parameters
    ----------
    ophys_workflow_app_config_path
        path to app config
    test_di_base_model_path
        dummy deepinterpolation base model path
    """
    os.environ['OPHYS_WORKFLOW_APP_CONFIG_PATH'] = \
        str(ophys_workflow_app_config_path)
    os.environ['TEST_DI_BASE_MODEL_PATH'] = str(test_di_base_model_path)

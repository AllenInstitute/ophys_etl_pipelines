import pytest
from ophys_etl.workflows.db.schemas import OphysROI


class TestOphysROI:

    @pytest.mark.parametrize("motion_border", [True, False])
    @pytest.mark.parametrize("decrosstalk_none", [True, False])
    def test_is_valid(self,
                      motion_border: bool,
                      decrosstalk_none: bool):
        """
        Test all possibilities for is_valid function in ROI.
        """
        mock_roi = OphysROI(
            id=1,
            x=0,
            y=0,
            width=2,
            height=1,
            is_in_motion_border=motion_border,
            is_decrosstalk_ghost=None if decrosstalk_none else False,
            is_decrosstalk_invalid_raw_active=False,
            is_decrosstalk_invalid_raw=False,
            is_decrosstalk_invalid_unmixed=False,
            is_decrosstalk_invalid_unmixed_active=False
        )

        if decrosstalk_none:
            with pytest.raises(TypeError, match=r"Decrosstalk flags not"):
                mock_roi.is_valid(equipment="MESO.1")
        else:
            assert mock_roi.is_valid(equipment="MESO.1") is not motion_border

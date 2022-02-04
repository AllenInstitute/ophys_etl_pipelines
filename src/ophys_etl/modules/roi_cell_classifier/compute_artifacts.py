import h5py
import json
from marshmallow import post_load
import argschema
import logging
import pathlib

from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi,
    get_roi_color_map)

from ophys_etl.modules.roi_cell_classifier.utils import (
    create_metadata,
    create_max_and_avg_projections,
    create_correlation_projection,
    get_traces)

from ophys_etl.utils.video_utils import (
    read_and_scale)


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class ArtifactFileSchema(argschema.ArgSchema):

    video_path = argschema.fields.InputFile(
            required=True,
            description=("Path to HDF5 file containing "
                         "video data on which to base this "
                         "artifact file"))

    roi_path = argschema.fields.InputFile(
            required=True,
            description=("Path to JSON file containing ROIs "
                         "on which to base this artifact file"))

    correlation_path = argschema.fields.InputFile(
            required=True,
            description=("Path to either png or pkl file which "
                         "contains correlation projection image"))

    artifact_path = argschema.fields.OutputFile(
            required=True,
            description=("Path to HDF5 artifact file to be written"))

    clobber = argschema.fields.Boolean(
            required=False,
            default=False,
            description=("If True, overwrite artifact path, if it exists"))

    video_lower_quantile = argschema.fields.Float(
            required=False,
            default=0.1,
            description=("lower quantile to use when scaling video data"))

    video_upper_quantile = argschema.fields.Float(
            required=False,
            default=0.999,
            description=("upper quantile to use when scaling video data"))

    projection_lower_quantile = argschema.fields.Float(
            required=False,
            default=0.2,
            description=("lower quantile to use when scaling "
                         "max/avg projection"))

    projection_upper_quantile = argschema.fields.Float(
            required=False,
            default=0.99,
            description=("upper quantile to use when scaling "
                         "max/avg projection"))

    @post_load
    def check_overwrite(self, data, **kwargs):
        output_path = pathlib.Path(data['artifact_path'])
        if output_path.exists():
            if not data['clobber']:
                msg = f'{output_path}\n exists.'
                msg += 'If you want to overwrite it, '
                msg += 'run with --clobber=True'
                raise RuntimeError(msg)

        correlation_suffix = pathlib.Path(data['correlation_path']).suffix
        if correlation_suffix not in set(['.png', '.pkl']):
            msg = "correlation_path must be .pkl or .png file;\n"
            msg += f"you gave:\n{data['correlation_path']}"
            raise RuntimeError(msg)

        return data


class ArtifactGenerator(argschema.ArgSchemaParser):

    default_schema = ArtifactFileSchema

    def run(self):
        video_path = pathlib.Path(self.args['video_path'])
        correlation_path = pathlib.Path(self.args['correlation_path'])
        roi_path = pathlib.Path(self.args['roi_path'])
        output_path = pathlib.Path(self.args['artifact_path'])

        with open(roi_path, 'rb') as in_file:
            raw_rois = json.load(in_file)
        extract_roi_list = sanitize_extract_roi_list(
                                raw_rois)
        ophys_roi_list = [extract_roi_to_ophys_roi(roi)
                          for roi in extract_roi_list]

        logger.info("read ROIs")

        trace_lookup = get_traces(video_path, ophys_roi_list)

        logger.info("wrote traces")

        metadata = create_metadata(
                         self.args,
                         video_path,
                         roi_path,
                         correlation_path)

        logger.info("hashed all input files")

        color_map = get_roi_color_map(ophys_roi_list)

        logger.info("computed color map")

        (avg_img_data,
         max_img_data) = create_max_and_avg_projections(
                             video_path,
                             self.args['projection_lower_quantile'],
                             self.args['projection_upper_quantile'])

        logger.info("Created max and avg projection images")

        correlation_img_data = create_correlation_projection(
                                    correlation_path)

        logger.info("Created correlation image")

        scaled_video = read_and_scale(
                          video_path=pathlib.Path(video_path),
                          origin=(0, 0),
                          frame_shape=max_img_data.shape,
                          quantiles=(self.args['video_lower_quantile'],
                                     self.args['video_upper_quantile']))

        logger.info("Created scaled video")

        with h5py.File(output_path, 'a') as out_file:
            out_file.create_dataset(
                'metadata',
                data=json.dumps(metadata, sort_keys=True).encode('utf-8'))
            out_file.create_dataset(
                'rois',
                data=json.dumps(extract_roi_list).encode('utf-8'))
            out_file.create_dataset(
                'roi_color_map',
                data=json.dumps(color_map).encode('utf-8'))
            out_file.create_dataset(
                'max_projection',
                data=max_img_data)
            out_file.create_dataset(
                'avg_projection',
                data=avg_img_data)
            out_file.create_dataset(
                'correlation_projection',
                data=correlation_img_data)
            out_file.create_dataset(
                'video_data',
                data=scaled_video)

            trace_group = out_file.create_group('traces')
            for roi_id in trace_lookup:
                trace_group.create_dataset(str(roi_id),
                                           data=trace_lookup[roi_id])

        logger.info("Wrote all artifacts")


if __name__ == "__main__":
    generator = ArtifactGenerator()
    generator.run()

import h5py
import json
from marshmallow import post_load
import copy
import argschema
import logging
import PIL.Image
import pathlib
import numpy as np

from ophys_etl.modules.segmentation.utils.roi_utils import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    get_roi_color_map)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.classifier2021.utils import (
    clip_img_to_quantiles,
    scale_img_to_uint8,
    get_traces,
    file_hash_from_path)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
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
            description=("upper quantil to use when scaling "
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

        metadata = dict()
        metadata['generator_args'] = copy.deepcopy(self.args)

        metadata['video'] = {
                'path': str(video_path.resolve().absolute()),
                'hash': file_hash_from_path(video_path)}

        metadata['rois'] = {
                'path': str(roi_path.resolve().absolute()),
                'hash': file_hash_from_path(roi_path)}

        metadata['correlation'] = {
                'path': str(correlation_path.resolve().absolute()),
                'hash': file_hash_from_path(correlation_path)}

        logger.info("hashed all input files")

        with open(roi_path, 'rb') as in_file:
            raw_rois = json.load(in_file)
        extract_roi_list = sanitize_extract_roi_list(
                                raw_rois)
        ophys_roi_list = [extract_roi_to_ophys_roi(roi)
                          for roi in extract_roi_list]

        logger.info("read ROIs")

        color_map = get_roi_color_map(ophys_roi_list)

        logger.info("computed color map")

        with h5py.File(video_path, 'r') as in_file:
            raw_video_data = in_file['data'][()]
        max_img_data = np.max(raw_video_data, axis=0)
        avg_img_data = np.mean(raw_video_data, axis=0)
        del raw_video_data

        max_img_data = clip_img_to_quantiles(
                           max_img_data,
                           (self.args['projection_lower_quantile'],
                            self.args['projection_upper_quantile']))

        max_img_data = scale_img_to_uint8(max_img_data)

        avg_img_data = clip_img_to_quantiles(
                           avg_img_data,
                           (self.args['projection_lower_quantile'],
                            self.args['projection_upper_quantile']))

        avg_img_data = scale_img_to_uint8(avg_img_data)

        logger.info("Created max and avg projection images")

        if self.args['correlation_path'].endswith('png'):
            correlation_img_data = np.array(
                                       PIL.Image.open(
                                           correlation_path, 'r'))
        else:
            correlation_img_data = graph_to_img(correlation_path)

        correlation_img_data = scale_img_to_uint8(correlation_img_data)

        logger.info("Created correlation image")

        scaled_video = read_and_scale(
                          video_path=pathlib.Path(video_path),
                          origin=(0, 0),
                          frame_shape=max_img_data.shape,
                          quantiles=(self.args['video_lower_quantile'],
                                     self.args['video_upper_quantile']))

        logger.info("Created scaled video")

        with h5py.File(output_path, 'w') as out_file:
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
                data=scaled_video,
                chunks=(100, 512, 512))

        logger.info("Wrote all data except traces")

        del scaled_video
        del max_img_data
        del avg_img_data
        del correlation_img_data

        trace_lookup = get_traces(video_path, ophys_roi_list)
        with h5py.File(output_path, 'a') as out_file:
            group = out_file.create_group('traces')
            for roi_id in trace_lookup:
                group.create_dataset(str(roi_id),
                                     data=trace_lookup[roi_id])


if __name__ == "__main__":
    generator = ArtifactGenerator()
    generator.run()
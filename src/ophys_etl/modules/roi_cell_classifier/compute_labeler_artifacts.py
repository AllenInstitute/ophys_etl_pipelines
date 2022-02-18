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

from ophys_etl.utils.motion_border import (
    get_max_correction_from_file,
    MotionBorder,
    motion_border_from_max_shift)


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class LabelerArtifactFileSchema(argschema.ArgSchema):

    video_path = argschema.fields.InputFile(
            required=True,
            description=("Path to HDF5 file containing "
                         "video data on which to base this "
                         "artifact file"))

    roi_path = argschema.fields.InputFile(
            required=True,
            description=("Path to JSON file containing ROIs "
                         "on which to base this artifact file"))

    motion_border_path = argschema.fields.InputFile(
            required=False,
            default=None,
            allow_none=True,
            description=("Path to csv file of rigid motion translations "
                         "applied to movie during motion correction; used "
                         "to calculate the motion border"))

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

        return data

    @post_load
    def check_file_types(self, data, **kwargs):
        msg = ''
        for file_path_key, suffix in [('video_path', '.h5'),
                                      ('roi_path', '.json'),
                                      ('motion_border_path', '.csv'),
                                      ('artifact_path', '.h5')]:
            file_path = data[file_path_key]
            if file_path is not None:
                file_path = pathlib.Path(file_path)
                if file_path.suffix != suffix:
                    msg += f'{file_path_key} must have suffix {suffix}; '
                    msg += f'you gave {str(file_path.resolve().absolute())}\n'
        file_path = pathlib.Path(data['correlation_path'])
        if file_path.suffix not in ('.pkl', '.png'):
            msg += 'correlation_path must have suffix either .pkl or .png; '
            msg += f'you gave {str(file_path.resolve().absolute())}\n'
        if len(msg) > 0:
            raise ValueError(msg)
        return data


class LabelerArtifactGenerator(argschema.ArgSchemaParser):

    default_schema = LabelerArtifactFileSchema

    def run(self):
        video_path = pathlib.Path(self.args['video_path'])
        correlation_path = pathlib.Path(self.args['correlation_path'])
        roi_path = pathlib.Path(self.args['roi_path'])
        if self.args['motion_border_path'] is not None:
            motion_border_path = pathlib.Path(self.args['motion_border_path'])
            max_shifts = get_max_correction_from_file(
                                   input_csv=motion_border_path)
            motion_border = motion_border_from_max_shift(max_shifts)
        else:
            motion_border_path = None
            motion_border = MotionBorder(left_side=0, right_side=0,
                                         top=0, bottom=0)

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
                         input_args=self.args,
                         video_path=video_path,
                         roi_path=roi_path,
                         correlation_path=correlation_path,
                         motion_csv_path=motion_border_path)

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

        # determine chunks in which to save the video data
        ntime = scaled_video.shape[0]
        nrows = scaled_video.shape[1]
        ncols = scaled_video.shape[2]
        video_chunks = (max(1, ntime//10),
                        max(1, nrows//16),
                        max(1, ncols//16))

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
                data=scaled_video,
                chunks=video_chunks)

            # note the transposition below;
            # if you shift up, the suspect pixels are those that wrap
            # on the bottom; if you shift right, the suspect pixels
            # are those that wrap on the right, etc.
            out_file.create_dataset(
                'motion_border',
                data=json.dumps({'bottom': motion_border.bottom,
                                 'top': motion_border.top,
                                 'left_side': motion_border.left_side,
                                 'right_side': motion_border.right_side},
                                indent=2).encode('utf-8'))

            trace_group = out_file.create_group('traces')
            for roi_id in trace_lookup:
                trace_group.create_dataset(str(roi_id),
                                           data=trace_lookup[roi_id])

        logger.info("Wrote all artifacts")


if __name__ == "__main__":
    generator = LabelerArtifactGenerator()
    generator.run()

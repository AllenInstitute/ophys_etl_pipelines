import datetime
import h5py
import json
import jsonlines
import os
from pathlib import Path
from typing import List, Tuple
import argschema
import imageio
import marshmallow as mm
from marshmallow.validate import OneOf
import numpy as np

from ophys_etl.rois import ROI
import ophys_etl.rois as ROIs
import ophys_etl.transforms.utils.video_utils as video_utils
import ophys_etl.transforms.utils.array_utils as array_utils
import ophys_etl.transforms.utils.image_utils as image_utils
from ophys_etl.schemas.fields import H5InputFile


class TransformPipelineSchema(argschema.ArgSchema):
    prod_segmentation_run_manifest = argschema.fields.InputFile(
        required=True,
        description=("A field which allows for a slapp manifest to be created "
                     "from a production segmentation run.")
    )
    output_manifest = argschema.fields.OutputFile(
        required=True,
        description="output path for jsonlines manifest contents")
    artifact_basedir = argschema.fields.OutputDir(
        required=True,
        description=("artifacts will be written to "
                     "artifact_basedir/segmentation_run_id."))
    # noinspection PyTypeChecker
    cropped_shape = argschema.fields.List(
        argschema.fields.Int,
        cli_as_single_argument=True,
        required=True,
        default=[128, 128],
        description="[h, w] of bounding box around ROI images/videos")
    outline_quantile = argschema.fields.Float(
        required=True,
        default=0.1,
        description="quantile threshold for outlining an ROI. ")
    input_fps = argschema.fields.Float(
        required=False,
        default=31.0,
        description="frames per second of input movie")
    output_fps = argschema.fields.Float(
        required=False,
        default=4.0,
        description="frames per second of downsampled movie")
    playback_factor = argschema.fields.Float(
        required=False,
        default=1.0,
        description=("webm FPS and trace pointInterval will adjust by this "
                     "factor relative to real time."))
    downsampling_strategy = argschema.fields.Str(
        required=False,
        default="average",
        validate=OneOf(['average', 'first', 'last', 'random']),
        description="what downsampling strategy to apply to movie and trace")
    random_seed = argschema.fields.Int(
        required=False,
        default=0,
        description="random seed to use if downsampling strategy is 'random'")
    movie_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.1,
        description=("lower quantile threshold for avg projection "
                     "histogram adjustment of movie"))
    movie_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.999,
        description=("upper quantile threshold for avg projection "
                     "histogram adjustment of movie"))
    projection_lower_quantile = argschema.fields.Float(
        required=False,
        default=0.2,
        description=("lower quantile threshold for projection "
                     "histogram adjustment"))
    projection_upper_quantile = argschema.fields.Float(
        required=False,
        default=0.99,
        description=("upper quantile threshold for projection "
                     "histogram adjustment"))
    webm_bitrate = argschema.fields.Str(
        required=False,
        default="0",
        description="passed as bitrate to imageio-ffmpeg.write_frames()")
    webm_quality = argschema.fields.Int(
        required=False,
        default=30,
        description=("Governs encoded video perceptual quality. "
                     "Can be from 0-63. Lower values mean higher quality. "
                     "Passed as crf to ffmpeg")
    )
    webm_parallelization = argschema.fields.Int(
        required=False,
        default=1,
        description=("Number of parallel processes to use for video encoding. "
                     "A value of -1 results in "
                     "using multiprocessing.cpu_count()")
    )
    scale_offset = argschema.fields.Int(
        required=False,
        default=3,
        description=("number of pixels scale corner is offset from "
                     "lower left in cropped field-of-view ROI outline"))
    full_scale_offset = argschema.fields.Int(
        required=False,
        default=12,
        description=("number of pixels scale corner is offset from "
                     "lower left in full field-of-view ROI outline"))
    scale_size_um = argschema.fields.Float(
        required=False,
        default=10.0,
        description=("length of scale bars in um in cropped field-of-view "
                     "ROI outline"))
    full_scale_size_um = argschema.fields.Float(
        required=False,
        default=40.0,
        description=("length of scale bars in um in full field-of-view "
                     "ROI outline"))
    um_per_pixel = argschema.fields.Float(
        required=False,
        default=0.78125,
        description="microns per pixel in the 2P source video")
    is_segmentation_labeling_app_input = argschema.fields.Bool(
        required=False,
        default=False,
        description="Includes relevant inputs if for segmentation labeling app (SLAPP) rather than for classifier")


class ProdSegmentationRunManifestSchema(mm.Schema):
    experiment_id = mm.fields.Int(required=True)
    binarized_rois_path = argschema.fields.InputFile(required=True)
    traces_h5_path = H5InputFile(required=False)
    movie_path = H5InputFile(required=True)
    local_to_global_roi_id_map = mm.fields.Dict(required=False,
                                                keys=mm.fields.Int(),
                                                values=mm.fields.Str())

    @mm.post_load
    def load_rois(self, data) -> dict:
        with open(data['binarized_rois_path'], 'r') as f:
            data['binarized_rois'] = json.load(f)
        return data

    @mm.post_load
    def get_movie_frame_shape(self, data, **kwargs) -> dict:
        with h5py.File(data['movie_path'], 'r') as h5f:
            data['movie_frame_shape'] = h5f['data'].shape[1:]
        return data


def xform_from_prod_manifest(prod_manifest_path: str) -> Tuple[List[ROI], Path]:
    with open(prod_manifest_path, 'r') as f:
        prod_manifest = json.load(f)
    prod_manifest = ProdSegmentationRunManifestSchema().load(prod_manifest)
    if 'local_to_global_roi_id_map' in prod_manifest:
        id_map = prod_manifest['local_to_global_roi_id_map']
        id_map = {int(k): v for k, v in id_map.items()}
    else:
        id_map = {roi['id']: roi['id'] for roi in prod_manifest['binarized_rois']}

    rois = []
    for roi in prod_manifest['binarized_rois']:
        if roi['id'] not in id_map:
            # only make manifests for listed ROIs
            continue

        roi_stamp = ROIs.coo_from_lims_style(
            mask_matrix=roi['mask_matrix'],
            xoffset=roi['x'],
            yoffset=roi['y'],
            shape=prod_manifest['movie_frame_shape'])
        converted_roi_id = id_map[roi['id']]

        roi_trace = None
        if 'traces_h5_path' in prod_manifest:
            with h5py.File(prod_manifest['traces_h5_path'], 'r') as h5f:
                traces_id_order = list(h5f['roi_names'][:].astype(int))
                roi_trace = h5f['data'][traces_id_order.index(roi['id'])]

        converted_roi = ROI(coo_rows=roi_stamp.row,
                            coo_cols=roi_stamp.col,
                            coo_data=roi_stamp.data.astype('uint8'),
                            image_shape=prod_manifest['movie_frame_shape'],
                            experiment_id=prod_manifest['experiment_id'],
                            roi_id=converted_roi_id,
                            trace=roi_trace,
                            is_binary=True)
        rois.append(converted_roi)

    return rois, Path(prod_manifest['movie_path'])


class TransformPipeline(argschema.ArgSchemaParser):
    default_schema = TransformPipelineSchema
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def run(self):
        rois, video_path = xform_from_prod_manifest(
            prod_manifest_path=self.args['prod_segmentation_run_manifest']
        )
        output_dir = Path(self.args['artifact_basedir']) / self.timestamp

        os.makedirs(output_dir, exist_ok=True)

        downsampled_video = video_utils.downsample_h5_video(
            video_path,
            self.args['input_fps'],
            self.args['output_fps'],
            self.args['downsampling_strategy'],
            self.args['random_seed'])

        # strategy for normalization: normalize entire video and projections
        # on quantiles of average projection before per-ROI processing
        movie_quantiles = [self.args['movie_lower_quantile'],
                           self.args['movie_upper_quantile']]
        proj_quantiles = [self.args['projection_lower_quantile'],
                          self.args['projection_upper_quantile']]

        max_projection = downsampled_video.max(axis=0)
        avg_projection = downsampled_video.mean(axis=0)

        # normalize movie according to avg quantiles
        downsampled_video = array_utils.normalize_array(array=downsampled_video, calc_cutoffs_from_array=avg_projection,
                                                        quantiles=movie_quantiles)

        # normalize avg projection
        avg_projection = array_utils.normalize_array(array=avg_projection, quantiles=proj_quantiles)

        # normalize max projection
        max_projection = array_utils.normalize_array(array=max_projection, quantiles=proj_quantiles)

        playback_fps = self.args['output_fps'] * self.args['playback_factor']

        # experiment-level artifact
        full_video_path = output_dir / "full_video.webm"
        if self.args['is_segmentation_labeling_app_input']:
            video_utils.transform_to_webm(
                video=downsampled_video, output_path=str(full_video_path),
                fps=playback_fps, ncpu=self.args['webm_parallelization'],
                bitrate=self.args['webm_bitrate'],
                crf=self.args['webm_quality'])

        # where to position the scales for the outlines
        scale_position = (
            self.args['scale_offset'],
            self.args['cropped_shape'][1] - self.args['scale_offset'])
        full_scale_position = (
            self.args['full_scale_offset'],
            downsampled_video.shape[2] - self.args['full_scale_offset'])

        # create the per-ROI artifacts
        insert_statements = []
        manifests = []
        for roi in rois:
            # mask and outline from ROI class
            mask_path = output_dir / f"mask_{roi.roi_id}.png"
            outline_path = output_dir / f"outline_{roi.roi_id}.png"
            full_outline_path = output_dir / f"full_outline_{roi.roi_id}.png"
            sub_video_path = output_dir / f"video_{roi.roi_id}.webm"
            max_proj_path = output_dir / f"max_{roi.roi_id}.png"
            avg_proj_path = output_dir / f"avg_{roi.roi_id}.png"
            trace_path = output_dir / f"trace_{roi.roi_id}.json"

            mask = roi.generate_ROI_mask(
                shape=self.args['cropped_shape'])
            mask = np.uint8(mask * 255 / mask.max())
            outline = roi.generate_ROI_outline(
                shape=self.args['cropped_shape'],
                quantile=self.args['outline_quantile'])
            full_outline = roi.generate_ROI_outline(
                shape=self.args['cropped_shape'],
                quantile=self.args['outline_quantile'],
                full=True)

            imageio.imsave(mask_path, mask, transparency=0)

            outline = image_utils.add_scale(
                outline,
                scale_position,
                self.args['um_per_pixel'],
                self.args['scale_size_um'],
                color=0,
                fontScale=0.3)
            full_outline = image_utils.add_scale(
                full_outline,
                full_scale_position,
                self.args['um_per_pixel'],
                self.args['full_scale_size_um'],
                color=0,
                thickness_um=1.5,
                fontScale=0.8)

            imageio.imsave(outline_path, outline, transparency=255)
            imageio.imsave(full_outline_path, full_outline, transparency=255)

            # video sub-frame
            inds, pads = array_utils.content_extents(
                roi._sparse_coo,
                shape=self.args['cropped_shape'],
                target_shape=tuple(downsampled_video.shape[1:]))
            sub_video = np.pad(
                downsampled_video[:, inds[0]:inds[1], inds[2]:inds[3]],
                ((0, 0), *pads))
            if self.args['is_segmentation_labeling_app_input']:
                video_utils.transform_to_webm(
                    video=sub_video, output_path=str(sub_video_path),
                    fps=playback_fps, ncpu=self.args['webm_parallelization'],
                    bitrate=self.args['webm_bitrate'],
                    crf=self.args['webm_quality'])

            # sub-projections
            sub_max = np.pad(
                max_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)

            sub_ave = np.pad(
                avg_projection[inds[0]:inds[1], inds[2]:inds[3]], pads)

            imageio.imsave(max_proj_path, sub_max)
            imageio.imsave(avg_proj_path, sub_ave)

            if self.args['is_segmentation_labeling_app_input']:
                # trace
                trace = array_utils.downsample_array(
                    np.array(roi.trace),
                    input_fps=self.args['input_fps'],
                    output_fps=self.args['output_fps'],
                    strategy=self.args['downsampling_strategy'],
                    random_seed=self.args['random_seed'])
                trace_json = {
                    "pointStart": 0,
                    "pointInterval": 1.0 / playback_fps,
                    "dataLength": len(trace),
                    "trace": trace.tolist()}
                with open(trace_path, "w") as fp:
                    json.dump(trace_json, fp)

            # manifest entry creation
            manifest = {'experiment-id': roi.experiment_id,
                        'roi-id': roi.roi_id,
                        'source-ref': str(outline_path),
                        'roi-mask-source-ref': str(mask_path),
                        'max-source-ref': str(max_proj_path),
                        'avg-source-ref': str(avg_proj_path),
                        'full-outline-source-ref': str(full_outline_path)}
            if self.args['is_segmentation_labeling_app_input']:
                manifest['trace-source-ref'] = str(trace_path)
                manifest['full-video-source-ref'] = str(full_video_path)
                manifest['video-source-ref'] = str(sub_video_path)

            manifests.append(manifest)

        with open(self.args['output_manifest'], "w") as fp:
            jsonlines.Writer(fp).write_all(manifests)


if __name__ == "__main__":  # pragma: no cover
    pipeline = TransformPipeline()
    pipeline.run()

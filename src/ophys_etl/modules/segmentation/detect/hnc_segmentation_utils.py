import hnccorr.base as hncbase
from typing import NamedTuple

from ophys_etl.modules.segmentation.modules.schemas import HNC_args


class HNCcorrSegmentationObjects(NamedTuple):
    # container for holding objects constructed out of the HNCcorr codebase
    SizePostprocessor: hncbase.SizePostprocessor
    HncParametricWrapper: hncbase.HncParametricWrapper
    PositiveSeedSelector: hncbase.PositiveSeedSelector
    NegativeSeedSelector: hncbase.NegativeSeedSelector
    GraphConstructor: hncbase.GraphConstructor
    Patch: hncbase.Patch
    CorrelationEmbedding: hncbase.CorrelationEmbedding
    patch_size: int


def hnc_construct(hnc_args: dict) -> HNCcorrSegmentationObjects:
    """
    construct HNCcorr objects given parameters

    Parameters
    ----------
    hnc_args: dict
        will be validated by HNC_args schema
    """

    # validate against schema
    args = HNC_args().load(data=hnc_args)

    # turn into an object
    config = hncbase.HNCcorrConfig(**args)

    edge_selector = hncbase.SparseComputationEmbeddingWrapper(
            config.sparse_computation_dimension,
            config.sparse_computation_grid_distance
        )
    hnc_obj = HNCcorrSegmentationObjects(
            SizePostprocessor=hncbase.SizePostprocessor(
                config.postprocessor_min_cell_size,
                config.postprocessor_max_cell_size,
                config.postprocessor_preferred_cell_size
            ),
            HncParametricWrapper=hncbase.HncParametricWrapper(0, 1),
            PositiveSeedSelector=hncbase.PositiveSeedSelector(
                config.positive_seed_radius),
            NegativeSeedSelector=hncbase.NegativeSeedSelector(
                config.negative_seed_circle_radius,
                config.negative_seed_circle_count
            ),
            GraphConstructor=hncbase.GraphConstructor(
                edge_selector,
                lambda a, b: hncbase.exponential_distance_decay(
                    a, b, config.gaussian_similarity_alpha)),
            Patch=hncbase.Patch,
            CorrelationEmbedding=hncbase.CorrelationEmbedding,
            patch_size=config.patch_size)
    return hnc_obj


class ExplicitHNCcorrCandidate:
    """like HNCcorr Candidate class, but, explicitly passing in objects
    and not the entire HNCcorr object
    """
    def __init__(self, center_seed, hnc_objs: HNCcorrSegmentationObjects):
        self.center_seed = center_seed
        self.hnc_objs = hnc_objs
        self.segmentations = None
        self.clean_segmentations = None
        self.best_segmentation = None

    def segment(self, movie):
        pos_seeds = self.hnc_objs.PositiveSeedSelector.select(
                self.center_seed, movie)
        neg_seeds = self.hnc_objs.NegativeSeedSelector.select(
                self.center_seed, movie)
        patch = self.hnc_objs.Patch(
            movie, self.center_seed, self.hnc_objs.patch_size)
        embedding = self.hnc_objs.CorrelationEmbedding(patch)
        graph = self.hnc_objs.GraphConstructor.construct(patch, embedding)
        self.segmentations = self.hnc_objs.HncParametricWrapper.solve(
                graph, pos_seeds, neg_seeds)
        self.clean_segmentations = [s.clean(pos_seeds, movie.pixel_shape)
                                    for s in self.segmentations]
        self.best_segmentation = self.hnc_objs.SizePostprocessor.select(
            self.clean_segmentations)
        return self.best_segmentation

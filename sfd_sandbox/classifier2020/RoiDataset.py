from PIL import Image
from croissant.utils import read_jsonlines
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from Transform import Transform
from deepcell_util import get_experiment_genotype_map


class RoiDataset(Dataset):
    def __init__(self, manifest_path, project_name, data_dir, image_dim=(128, 128), roi_ids=None,
                 transform: Transform = None, debug=False, has_labels=True, parse_from_manifest=True,
                 cre_line=None, exclude_mask=False):
        super().__init__()

        if not parse_from_manifest and roi_ids is None:
            raise ValueError('need to provide roi ids if not parsing from manifest')

        self.manifest_path = manifest_path
        self.project_name = project_name
        self.data_dir = data_dir
        self.image_dim = image_dim
        self.transform = transform
        self.has_labels = has_labels
        self.exclude_mask = exclude_mask

        experiment_genotype_map = get_experiment_genotype_map()

        if parse_from_manifest:
            manifest = read_jsonlines(uri=self.manifest_path)
            self.manifest = [x for x in manifest]
        else:
            self.manifest = [{'roi-id': roi_id} for roi_id in roi_ids]

        if cre_line:
            self.manifest = self._filter_by_cre_line(experiment_genotype_map=experiment_genotype_map, cre_line=cre_line)

        if roi_ids is not None:
            self.manifest = [x for x in self.manifest if str(x['roi-id']) in set([str(x) for x in roi_ids])]
            self.roi_ids = [x['roi-id'] for x in self.manifest]
        else:
            self.roi_ids = [x['roi-id'] for x in self.manifest]

        self.y = self._get_labels() if self.has_labels else None

        if parse_from_manifest:
            self.cre_line = self._get_creline(experiment_genotype_map=experiment_genotype_map)
        else:
            self.cre_line = None

        if debug:
            not_cell_idx = np.argwhere(self.y == 0)[0][0]
            cell_idx = np.argwhere(self.y == 1)[0][0]
            self.manifest = [self.manifest[not_cell_idx], self.manifest[cell_idx]]
            self.y = np.array([0, 1])
            self.roi_ids = [x['roi-id'] for x in self.manifest]

    def __getitem__(self, index):
        obs = self.manifest[index]

        input = self._extract_channels(obs=obs)

        if self.transform:
            avg, max_, mask = input[:, :, 0], input[:, :, 1], input[:, :, 2]
            if self.transform.avg_transform:
                avg = self.transform.avg_transform(avg)
                input[:, :, 0] = avg
            if self.transform.max_transform:
                max_ = self.transform.max_transform(max_)
                input[:, :, 1] = max_
            if self.transform.mask_transform:
                mask = self.transform.mask_transform(mask)
                input[:, :, 2] = mask

            if self.transform.all_transform:
                input = self.transform.all_transform(input)

        # Note if no labels, 0.0 is given as the target
        # TODO collate_fn should be used instead
        target = self.y[index] if self.has_labels else 0

        return input, target

    def __len__(self):
        return len(self.manifest)

    def _get_labels(self):
        labels = [x[self.project_name]['majorityLabel'] for x in self.manifest]
        labels = [int(x == 'cell') for x in labels]
        labels = np.array(labels)
        return labels

    def _get_creline(self, experiment_genotype_map):
        cre_lines = []
        for x in self.manifest:
            if 'cre_line' in x:
                cre_lines.append(x['cre_line'])
            else:
                cre_lines.append(experiment_genotype_map[x['experiment-id']][:3])
        return cre_lines

    def _extract_channels(self, obs):
        roi_id = obs['roi-id']

        data_dir = self.data_dir
        with open(f'{data_dir}/avg_{roi_id}.png', 'rb') as f:
            avg = Image.open(f)
            avg = np.array(avg)

        with open(f'{data_dir}/max_{roi_id}.png', 'rb') as f:
            max = Image.open(f)
            max = np.array(max)

        with open(f'{data_dir}/mask_{roi_id}.png', 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)

        res = np.zeros((*self.image_dim, 3), dtype=np.uint8)
        res[:, :, 0] = avg
        res[:, :, 1] = max

        if self.exclude_mask:
            res[:, :, 2] = max
        else:
            try:
                res[:, :, 2] = mask
            except:
                # TODO this should be fixed in SLAPP (issue: ROI larger than cropped area)
                pass

        return res

    def _filter_by_cre_line(self, experiment_genotype_map, cre_line):
        filtered = [x for x in self.manifest if experiment_genotype_map[x['experiment-id']].startswith(cre_line)]
        return filtered


class SlcSampler(Sampler):
    def __init__(self, y, cell_prob=0.2):
        self.positive_proba = cell_prob

        self.positive_idxs = np.where(y == 1)[0]
        self.negative_idxs = np.where(y == 0)[0]

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative


if __name__ == '__main__':
    project_name = 'ophys-experts-slc-oct-2020_ophys-experts-go-big-or-go-home'
    manifest_path = 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020_behavior_3cre_1600roi_merged/output.manifest'

    dataset = RoiDataset(manifest_path=manifest_path, project_name=project_name, data_dir='./data')


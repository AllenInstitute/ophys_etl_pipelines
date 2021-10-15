class Transform:
    def __init__(self, all_transform=None, mask_transform=None, max_transform=None, avg_transform=None):
        self.all_transform = all_transform
        self.mask_transform = mask_transform
        self.max_transform = max_transform
        self.avg_transform = avg_transform
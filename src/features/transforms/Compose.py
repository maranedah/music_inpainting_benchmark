class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, midi):
        for t in self.transforms:
            midi = t(midi)
        return midi
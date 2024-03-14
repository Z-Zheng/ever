from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image, target)
            return image

        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class CustomOp(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, image, target=None):
        if target is None:
            return self.fn(image)
        return self.fn(image, target)


class THMeanStdNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

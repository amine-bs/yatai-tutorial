import random
import os
from PIL import Image
import torchvision.transforms.functional as TF
import boto3
from torch.utils.data import Dataset


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(key, bucket="mbenxsalha"):
    s3 = boto3.client('s3', endpoint_url='https://minio.lab.sspcloud.fr/')
    data = s3.get_object(Bucket=bucket, Key=key)
    img = Image.open(data["Body"]).convert("RGB")
    return img


def make_dataset(root="diffusion/pizza-not-pizza"):
    s3 = boto3.client('s3',endpoint_url='https://minio.lab.sspcloud.fr/')
    data = []

    pizza_root = os.path.join(root, "pizza")
    for img in s3.list_objects(Bucket="mbenxsalha", Prefix=pizza_root)["Contents"]:
        key = img["Key"]
        if is_image_file(key):
            data.append((key, 1))

        not_pizza_root = os.path.join(root, "not_pizza")
        for img in s3.list_objects(Bucket="mbenxsalha", Prefix=not_pizza_root)["Contents"]:
            key = img["Key"]
            if is_image_file(key):
                data.append((key, 0))
    return data


class MyTransformer():
    def __init__(self, crop):
        self.crop = crop

    def __call__(self, img, rot=None):
        img = TF.resize(img, (256, 256))
        img = TF.crop(img, self.crop[0], self.crop[1], 224, 224)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img


class DatasetGenerator(Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, lab = self.imgs[index]
        img = load_image(path)
        # If a custom transform is specified apply that transform
        if self.transform is not None:
            img = self.transform(img)
        else:  # Otherwise define a random one (random cropping)
            top = random.randint(0, 256 - 224)
            left = random.randint(0, 256 - 224)
            transform = MyTransformer([top, left])
            # Apply the transformation
            img = transform(img)
        return img, lab

    def __len__(self):
        return len(self.imgs)
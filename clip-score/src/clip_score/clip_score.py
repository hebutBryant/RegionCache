"""Calculates the CLIP Scores

The CLIP model is a contrasitively learned language-image model. There is
an image encoder and a text encoder. It is believed that the CLIP model could
measure the similarity of cross modalities. Please find more information from
https://github.com/openai/CLIP.

The CLIP Score measures the Cosine Similarity between two embedded features.
This repository utilizes the pretrained CLIP Model to calculate
the mean average of cosine similarities.

See --help to see further details.

Code adapted from https://github.com/mseitzer/pytorch-fid and
https://github.com/openai/CLIP.

Copyright 2025 The Chinese University of Hong Kong, Shenzhen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size',
                    type=int,
                    default=50,
                    help='Batch size to use')
parser.add_argument('--clip-model',
                    type=str,
                    default='openai/clip-vit-base-patch16',
                    help='CLIP model to use')
parser.add_argument('--num-workers',
                    type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device',
                    type=str,
                    default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--real_flag',
                    type=str,
                    default='img',
                    help=('The modality of real path. '
                          'Default to img'))
parser.add_argument('--fake_flag',
                    type=str,
                    default='txt',
                    help=('The modality of real path. '
                          'Default to txt'))
parser.add_argument('real_path', type=str, help=('Path to your modality data'))
parser.add_argument('fake_path', type=str, help=('Path to your modality data'))

IMAGE_EXTENSIONS = {
    'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'
}

TEXT_EXTENSIONS = {'txt'}


class DummyDataset(Dataset):

    FLAGS = ['img', 'txt']

    def __init__(self,
                 real_path,
                 fake_path,
                 real_flag: str = 'img',
                 fake_flag: str = 'txt',
                 transform=None,
                 tokenizer=None) -> None:
        super().__init__()
        if real_flag not in self.FLAGS or fake_flag not in self.FLAGS:
            raise TypeError('CLIP Score only support modality of {}. '
                            'However, get {} and {}'.format(
                                self.FLAGS, real_flag, fake_flag))
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_folder = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        if isinstance(self.real_folder, list):
            real_folder_lenghth = len(self.real_folder)
        else:
            real_folder_lenghth = 1
        if isinstance(self.fake_folder, list):
            fake_folder_lenghth = len(self.fake_folder)
        else:
            fake_folder_lenghth = 1
        return max(real_folder_lenghth, fake_folder_lenghth)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        if isinstance(self.real_folder, list):
            real_path = self.real_folder[index]
        else:
            real_path = self.real_folder
        if isinstance(self.fake_folder, list):
            fake_path = self.fake_folder[index]
        else:
            fake_path = self.fake_folder
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample

    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError('Got unexpected modality: {}'.format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(text=None, images=img)
            img['pixel_values'] = img['pixel_values'][0]
        return img

    def _load_txt(self, path):
        if osp.exists(path):
            with open(path, 'r') as fp:
                data = fp.read()
                fp.close()
        else:
            data = path
        if self.transform is not None:
            data = self.tokenizer(data, padding=True, return_tensors='pt')
            for key in data:
                data[key] = data[key].squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True

    def _combine_without_prefix(self, folder_path, prefix='.'):
        if not osp.exists(folder_path):
            return folder_path
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    for batch_data in tqdm(dataloader):
        real = batch_data['real']
        real_features = forward_modality(model, real, real_flag)
        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)

        # normalize features
        real_features = real_features / real_features.norm(
            dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(
            dim=1, keepdim=True).to(torch.float32)

        # calculate scores
        score = (fake_features * real_features).sum()
        score_acc += score
        sample_num += real_features.shape[0]

    return score_acc / sample_num


def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    for key in data:
        data[key] = data[key].to(device)
    if flag == 'img':
        features = model.get_image_features(**data)
    elif flag == 'txt':
        features = model.get_text_features(**data)
    else:
        raise TypeError
    return features


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    print('Loading CLIP model: {}'.format(args.clip_model))
    model = AutoModel.from_pretrained(args.clip_model).to(device)
    processor = AutoProcessor.from_pretrained(args.clip_model)
    tokenizer = AutoTokenizer.from_pretrained(args.clip_model)

    dataset = DummyDataset(args.real_path,
                           args.fake_path,
                           args.real_flag,
                           args.fake_flag,
                           transform=processor,
                           tokenizer=tokenizer)
    dataloader = DataLoader(dataset,
                            args.batch_size,
                            num_workers=num_workers,
                            pin_memory=True)

    print('Calculating CLIP Score:')
    clip_score = calculate_clip_score(dataloader, model, args.real_flag,
                                      args.fake_flag)
    clip_score = clip_score.cpu().item()
    print('CLIP Score: ', clip_score)


if __name__ == '__main__':
    main()

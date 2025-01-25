import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

import os
class hut290Dataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.hut290_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls= self.sequence_list[i]
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        target_path = '{}/{}'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        sum = count_files(target_path)
        occlusion_label_path = '{}/{}/full_occlusion.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        #full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        #out_of_view_label_path = '{}/{}/out_of_view.txt'.format(self.base_path,  sequence_name)
        #out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        #target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, sum  - 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'hut290', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=None)

    def __len__(self):
        return len(self.sequence_list)


    def _get_sequence_list(self):
            sequence_list = [
                'dolphin7',
                'fish59',
                'fish63',
                'dolphin6',
                'fish7',
                'fish11',
                'fish17',
                'fish18',
                'fish21',
                'fish26',
                'fish28',
                'fish29',
                'fish32',
                'fish35',
                'cuttlefish1',
                'crab1',
                'dolphin2',
                'fish38',
                'fish44',
                'fish48',
                'fish53',
                'fish57',
                'fish58',
                'fish61',
                'fish65',
                'fish139',
                'fish142',
                'fish146',
                'fish145',
                'fish148',
                'fish149',
                'fish154',
                'fish157',
                'fish158',
                'fish159',
                'fish160',
                'fish170',
                'fish171',
                'fish173',
                'fish177',
                'fish181',
                'fish186',
                'fish187',
                'octopus1',
                'seahorse1',
                'plant1',
                'unknown10',
                'unknown12',
                'unknown16',
                'unknown19',
                'fish67',
                'fish69',
                'fish70',
                'fish75',
                'fish78',
                'fish81',
                'fish88',
                'fish90',
                'fish95',
                'fish99',
                'fish101',
                'fish103',
                'fish110',
                'fish116',
                'fish119',
                'fish121',
                'fish134',
                'goldfish',
                'human8',
                'human9',
                'human12',
                'jellyfish2',
                'plankton1',
                'seaturtle6',
                'seaturtle9',
                'seaturtle16',
                'shark2',
                'unknown1',
                'unknown4',
                'unknown6',
                'fish122',
                'fish1',
                'fish51',
                'fish54',
                'fish55',
                'fish83',
                'fish86',
                'fish100',
                'fish105',
                'fish108'

            ]
            return sequence_list

def count_files(path):
    count = 0

    for root, dirs, files in os.walk(path):
        count += len(files)

    return count




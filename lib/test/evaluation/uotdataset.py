import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from ..utils.load_text import load_text


class UOTDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uot_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(' ', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'utb', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {
                "name": "CenoteAngelita",
                "path": "CenoteAngelita/img",
                "startFrame": 1,
                "endFrame": 1107,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CenoteAngelita/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MantaRescue2",
                "path": "MantaRescue2/img",
                "startFrame": 1,
                "endFrame": 1332,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MantaRescue2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FightingEels1",
                "path": "FightingEels1/img",
                "startFrame": 1,
                "endFrame": 753,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FightingEels1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "HoverFish2",
                "path": "HoverFish2/img",
                "startFrame": 1,
                "endFrame": 449,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "HoverFish2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Submarine",
                "path": "Submarine/img",
                "startFrame": 1,
                "endFrame": 408,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Submarine/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Diving360Degree3",
                "path": "Diving360Degree3/img",
                "startFrame": 1,
                "endFrame": 422,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Diving360Degree3/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GreenMoreyEel2",
                "path": "GreenMoreyEel2/img",
                "startFrame": 1,
                "endFrame": 687,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GreenMoreyEel2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Ballena",
                "path": "Ballena/img",
                "startFrame": 1,
                "endFrame": 910,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Ballena/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CressiGuillaumeNeri1",
                "path": "CressiGuillaumeNeri1/img",
                "startFrame": 1,
                "endFrame": 641,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CressiGuillaumeNeri1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Fisherman",
                "path": "Fisherman/img",
                "startFrame": 1,
                "endFrame": 708,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Fisherman/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "HoverFish1",
                "path": "HoverFish1/img",
                "startFrame": 1,
                "endFrame": 673,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "HoverFish1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SeaDragon",
                "path": "SeaDragon/img",
                "startFrame": 1,
                "endFrame": 944,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SeaDragon/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ArmyDiver2",
                "path": "ArmyDiver2/img",
                "startFrame": 1,
                "endFrame": 525,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ArmyDiver2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Rocketman",
                "path": "Rocketman/img",
                "startFrame": 1,
                "endFrame": 463,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Rocketman/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "OceanFloorSensor",
                "path": "OceanFloorSensor/img",
                "startFrame": 1,
                "endFrame": 309,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "OceanFloorSensor/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "WhaleDiving",
                "path": "WhaleDiving/img",
                "startFrame": 1,
                "endFrame": 480,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "WhaleDiving/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SofiaRocks2",
                "path": "SofiaRocks2/img",
                "startFrame": 1,
                "endFrame": 453,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SofiaRocks2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FishingAdventure",
                "path": "FishingAdventure/img",
                "startFrame": 1,
                "endFrame": 630,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FishingAdventure/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Cuttlefish",
                "path": "Cuttlefish/img",
                "startFrame": 1,
                "endFrame": 279,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Cuttlefish/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "RedSeaReptile",
                "path": "RedSeaReptile/img",
                "startFrame": 1,
                "endFrame": 1130,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "RedSeaReptile/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MantisShrimp",
                "path": "MantisShrimp/img",
                "startFrame": 1,
                "endFrame": 862,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MantisShrimp/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SeaTurtle3",
                "path": "SeaTurtle3/img",
                "startFrame": 1,
                "endFrame": 878,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SeaTurtle3/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FlukeFishing2",
                "path": "FlukeFishing2/img",
                "startFrame": 1,
                "endFrame": 672,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FlukeFishing2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "DefenseInTheSea2",
                "path": "DefenseInTheSea2/img",
                "startFrame": 1,
                "endFrame": 451,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "DefenseInTheSea2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MantaRescue1",
                "path": "MantaRescue1/img",
                "startFrame": 1,
                "endFrame": 624,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MantaRescue1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ArmyDiver1",
                "path": "ArmyDiver1/img",
                "startFrame": 1,
                "endFrame": 684,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ArmyDiver1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Kleptopus1",
                "path": "Kleptopus1/img",
                "startFrame": 1,
                "endFrame": 838,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Kleptopus1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "HappyTurtle1",
                "path": "HappyTurtle1/img",
                "startFrame": 1,
                "endFrame": 1404,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "HappyTurtle1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GarryFish",
                "path": "GarryFish/img",
                "startFrame": 1,
                "endFrame": 472,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GarryFish/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "HeartShape",
                "path": "HeartShape/img",
                "startFrame": 1,
                "endFrame": 577,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "HeartShape/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Lobsters2",
                "path": "Lobsters2/img",
                "startFrame": 1,
                "endFrame": 1053,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Lobsters2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Dolphin2",
                "path": "Dolphin2/img",
                "startFrame": 1,
                "endFrame": 397,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Dolphin2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "BlueFish2",
                "path": "BlueFish2/img",
                "startFrame": 1,
                "endFrame": 593,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "BlueFish2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FreeDiver2",
                "path": "FreeDiver2/img",
                "startFrame": 1,
                "endFrame": 255,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FreeDiver2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GuillaumeNery",
                "path": "GuillaumeNery/img",
                "startFrame": 1,
                "endFrame": 337,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GuillaumeNery/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "LargemouthBass",
                "path": "LargemouthBass/img",
                "startFrame": 1,
                "endFrame": 626,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "LargemouthBass/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Diving360Degree1",
                "path": "Diving360Degree1/img",
                "startFrame": 1,
                "endFrame": 443,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Diving360Degree1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Kleptopus2",
                "path": "Kleptopus2/img",
                "startFrame": 1,
                "endFrame": 811,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Kleptopus2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SharkSuckers2",
                "path": "SharkSuckers2/img",
                "startFrame": 1,
                "endFrame": 817,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SharkSuckers2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Steinlager",
                "path": "Steinlager/img",
                "startFrame": 1,
                "endFrame": 283,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Steinlager/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ThePassage",
                "path": "ThePassage/img",
                "startFrame": 1,
                "endFrame": 283,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ThePassage/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ScubaDiving2",
                "path": "ScubaDiving2/img",
                "startFrame": 1,
                "endFrame": 505,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ScubaDiving2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MonsterCreature1",
                "path": "MonsterCreature1/img",
                "startFrame": 1,
                "endFrame": 823,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MonsterCreature1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "LittleMonster",
                "path": "LittleMonster/img",
                "startFrame": 1,
                "endFrame": 582,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "LittleMonster/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MonsterCreature2",
                "path": "MonsterCreature2/img",
                "startFrame": 1,
                "endFrame": 960,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MonsterCreature2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SeaTurtle2",
                "path": "SeaTurtle2/img",
                "startFrame": 1,
                "endFrame": 823,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SeaTurtle2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CressiGuillaumeNeri2",
                "path": "CressiGuillaumeNeri2/img",
                "startFrame": 1,
                "endFrame": 1064,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CressiGuillaumeNeri2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SeaDiver",
                "path": "SeaDiver/img",
                "startFrame": 1,
                "endFrame": 818,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SeaDiver/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "WolfTrolling",
                "path": "WolfTrolling/img",
                "startFrame": 1,
                "endFrame": 396,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "WolfTrolling/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "BallisticMissile2",
                "path": "BallisticMissile2/img",
                "startFrame": 1,
                "endFrame": 468,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "BallisticMissile2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MuckySecrets1",
                "path": "MuckySecrets1/img",
                "startFrame": 1,
                "endFrame": 362,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MuckySecrets1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SharkSuckers1",
                "path": "SharkSuckers1/img",
                "startFrame": 1,
                "endFrame": 381,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SharkSuckers1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SeaTurtle1",
                "path": "SeaTurtle1/img",
                "startFrame": 1,
                "endFrame": 875,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SeaTurtle1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CleverOctopus",
                "path": "CleverOctopus/img",
                "startFrame": 1,
                "endFrame": 1049,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CleverOctopus/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Octopus2",
                "path": "Octopus2/img",
                "startFrame": 1,
                "endFrame": 1573,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Octopus2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CoconutOctopus2",
                "path": "CoconutOctopus2/img",
                "startFrame": 1,
                "endFrame": 606,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CoconutOctopus2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "DefenseInTheSea1",
                "path": "DefenseInTheSea1/img",
                "startFrame": 1,
                "endFrame": 475,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "DefenseInTheSea1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ColourChangingSquid",
                "path": "ColourChangingSquid/img",
                "startFrame": 1,
                "endFrame": 315,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ColourChangingSquid/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "HappyTurtle2",
                "path": "HappyTurtle2/img",
                "startFrame": 1,
                "endFrame": 762,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "HappyTurtle2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "DeepSeaFish",
                "path": "DeepSeaFish/img",
                "startFrame": 1,
                "endFrame": 1186,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "DeepSeaFish/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MantaRescue3",
                "path": "MantaRescue3/img",
                "startFrame": 1,
                "endFrame": 1127,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MantaRescue3/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Dolphin1",
                "path": "Dolphin1/img",
                "startFrame": 1,
                "endFrame": 1078,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Dolphin1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ArmyDiver3",
                "path": "ArmyDiver3/img",
                "startFrame": 1,
                "endFrame": 931,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ArmyDiver3/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MississippiFish",
                "path": "MississippiFish/img",
                "startFrame": 1,
                "endFrame": 666,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MississippiFish/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FightingEels2",
                "path": "FightingEels2/img",
                "startFrame": 1,
                "endFrame": 619,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FightingEels2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CoralGardenSea2",
                "path": "CoralGardenSea2/img",
                "startFrame": 1,
                "endFrame": 398,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CoralGardenSea2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "JerkbaitBites",
                "path": "JerkbaitBites/img",
                "startFrame": 1,
                "endFrame": 516,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "JerkbaitBites/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GiantCuttlefish2",
                "path": "GiantCuttlefish2/img",
                "startFrame": 1,
                "endFrame": 502,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GiantCuttlefish2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MantaRescue4",
                "path": "MantaRescue4/img",
                "startFrame": 1,
                "endFrame": 790,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MantaRescue4/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FreeDiver1",
                "path": "FreeDiver1/img",
                "startFrame": 1,
                "endFrame": 448,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FreeDiver1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MuckySecrets2",
                "path": "MuckySecrets2/img",
                "startFrame": 1,
                "endFrame": 376,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MuckySecrets2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CrayFish",
                "path": "CrayFish/img",
                "startFrame": 1,
                "endFrame": 507,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CrayFish/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FlukeFishing1",
                "path": "FlukeFishing1/img",
                "startFrame": 1,
                "endFrame": 1324,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FlukeFishing1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Skagerrak",
                "path": "Skagerrak/img",
                "startFrame": 1,
                "endFrame": 313,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Skagerrak/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CoconutOctopus1",
                "path": "CoconutOctopus1/img",
                "startFrame": 1,
                "endFrame": 548,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CoconutOctopus1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "WhaleAtBeach2",
                "path": "WhaleAtBeach2/img",
                "startFrame": 1,
                "endFrame": 317,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "WhaleAtBeach2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "BoySwimming",
                "path": "BoySwimming/img",
                "startFrame": 1,
                "endFrame": 648,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "BoySwimming/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "PinkFish",
                "path": "PinkFish/img",
                "startFrame": 1,
                "endFrame": 417,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "PinkFish/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "PlayingTurtle",
                "path": "PlayingTurtle/img",
                "startFrame": 1,
                "endFrame": 423,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "PlayingTurtle/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "AntiguaTurtle",
                "path": "AntiguaTurtle/img",
                "startFrame": 1,
                "endFrame": 971,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "AntiguaTurtle/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Diving360Degree2",
                "path": "Diving360Degree2/img",
                "startFrame": 1,
                "endFrame": 458,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Diving360Degree2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "WhiteShark",
                "path": "WhiteShark/img",
                "startFrame": 1,
                "endFrame": 562,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "WhiteShark/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "DeepOceanLostWorld",
                "path": "DeepOceanLostWorld/img",
                "startFrame": 1,
                "endFrame": 981,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "DeepOceanLostWorld/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SharkCloseCall1",
                "path": "SharkCloseCall1/img",
                "startFrame": 1,
                "endFrame": 538,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SharkCloseCall1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ScubaDiving1",
                "path": "ScubaDiving1/img",
                "startFrame": 1,
                "endFrame": 662,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ScubaDiving1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "WhaleAtBeach1",
                "path": "WhaleAtBeach1/img",
                "startFrame": 1,
                "endFrame": 1353,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "WhaleAtBeach1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FightToDeath",
                "path": "FightToDeath/img",
                "startFrame": 1,
                "endFrame": 675,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FightToDeath/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GreenMoreyEel3",
                "path": "GreenMoreyEel3/img",
                "startFrame": 1,
                "endFrame": 699,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GreenMoreyEel3/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ClickerAndTarget",
                "path": "ClickerAndTarget/img",
                "startFrame": 1,
                "endFrame": 1764,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ClickerAndTarget/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CoralGardenSea1",
                "path": "CoralGardenSea1/img",
                "startFrame": 1,
                "endFrame": 258,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CoralGardenSea1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SharkCloseCall2",
                "path": "SharkCloseCall2/img",
                "startFrame": 1,
                "endFrame": 719,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SharkCloseCall2/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "SofiaRocks1",
                "path": "SofiaRocks1/img",
                "startFrame": 1,
                "endFrame": 415,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "SofiaRocks1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Lobsters1",
                "path": "Lobsters1/img",
                "startFrame": 1,
                "endFrame": 412,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Lobsters1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FishingBait",
                "path": "FishingBait/img",
                "startFrame": 1,
                "endFrame": 654,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FishingBait/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "MythBusters",
                "path": "MythBusters/img",
                "startFrame": 1,
                "endFrame": 1095,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "MythBusters/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GiantCuttlefish1",
                "path": "GiantCuttlefish1/img",
                "startFrame": 1,
                "endFrame": 714,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GiantCuttlefish1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "FishFollowing",
                "path": "FishFollowing/img",
                "startFrame": 1,
                "endFrame": 1670,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "FishFollowing/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "Octopus1",
                "path": "Octopus1/img",
                "startFrame": 1,
                "endFrame": 1662,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "Octopus1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "CrabTrap",
                "path": "CrabTrap/img",
                "startFrame": 1,
                "endFrame": 827,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "CrabTrap/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "BlueFish1",
                "path": "BlueFish1/img",
                "startFrame": 1,
                "endFrame": 759,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "BlueFish1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "WallEye",
                "path": "WallEye/img",
                "startFrame": 1,
                "endFrame": 600,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "WallEye/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "GreenMoreyEel1",
                "path": "GreenMoreyEel1/img",
                "startFrame": 1,
                "endFrame": 1056,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "GreenMoreyEel1/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "ElephantSeals",
                "path": "ElephantSeals/img",
                "startFrame": 1,
                "endFrame": 438,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "ElephantSeals/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "HappyTurtle3",
                "path": "HappyTurtle3/img",
                "startFrame": 1,
                "endFrame": 612,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "HappyTurtle3/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "NeryClimbing",
                "path": "NeryClimbing/img",
                "startFrame": 1,
                "endFrame": 313,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "NeryClimbing/groundtruth_rect.txt",
                "object_class": "person"
            },
            {
                "name": "BallisticMissile1",
                "path": "BallisticMissile1/img",
                "startFrame": 1,
                "endFrame": 516,
                "nz": 4,
                "ext": "jpg",
                "anno_path": "BallisticMissile1/groundtruth_rect.txt",
                "object_class": "person"
            }
        ]

        return sequence_info_list

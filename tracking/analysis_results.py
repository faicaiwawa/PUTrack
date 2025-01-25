import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'hut290'

trackers.extend(trackerlist(name='siamrpn++', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='siamrpn++'))

trackers.extend(trackerlist(name='aiatrack', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='aiatrack'))

trackers.extend(trackerlist(name='automatch', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='automatch'))

trackers.extend(trackerlist(name='mixformer2_vit_online', parameter_name='288_depth8_score', dataset_name=dataset_name,
                            run_ids=None, display_name='mixformer2_vit_online'))

trackers.extend(trackerlist(name='mixformer_cvt_online', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='mixformer_cvt_online'))

trackers.extend(trackerlist(name='odtrack', parameter_name='baseline_300', dataset_name=dataset_name,
                            run_ids=None, display_name='odtrack'))

trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='ostrack'))

trackers.extend(trackerlist(name='cswintt', parameter_name='baseline_cs_000', dataset_name=dataset_name,
                            run_ids=None, display_name='cswintt'))

trackers.extend(trackerlist(name='simtrack', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='simtrack'))

trackers.extend(trackerlist(name='grm', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='grm'))

trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b256', dataset_name=dataset_name,
                            run_ids=None, display_name='seqtrack'))

trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='stark_st'))

trackers.extend(trackerlist(name='transt', parameter_name='transt50', dataset_name=dataset_name,
                            run_ids=None, display_name='transt'))

trackers.extend(trackerlist(name='siamfc', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='siamfc'))

trackers.extend(trackerlist(name='siamfc++', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='siamfc++'))

trackers.extend(trackerlist(name='atom', parameter_name='default_000', dataset_name=dataset_name,
                            run_ids=None, display_name='atom'))

trackers.extend(trackerlist(name='dimp', parameter_name='dimp50_000', dataset_name=dataset_name,
                            run_ids=None, display_name='dimp50'))

trackers.extend(trackerlist(name='siamban', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='siamban'))

trackers.extend(trackerlist(name='stmtrack', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='stmtrack'))

trackers.extend(trackerlist(name='keep_track', parameter_name='default_000', dataset_name=dataset_name,
                            run_ids=None, display_name='keep_track'))

trackers.extend(trackerlist(name='tomp', parameter_name='tomp50_000', dataset_name=dataset_name,
                            run_ids=None, display_name='tomp'))

# raw results of this tracker only  contain hut290, results of uot and utb you can find in its paper.
# trackers.extend(trackerlist(name='uostrack', parameter_name='wrfish', dataset_name=dataset_name,
#                              run_ids=None, display_name='uostrack'))

trackers.extend(trackerlist(name='PU-OSTrack', parameter_name='baseline', dataset_name=dataset_name,
                            run_ids=None, display_name='PU-OSTrack'))

trackers.extend(trackerlist(name='artrack_seq', parameter_name='artrack_seq_256_full', dataset_name=dataset_name,
                            run_ids=None, display_name='ARTrack_seq'))

trackers.extend(trackerlist(name='PU-Artrack_seq', parameter_name='PU-Artrack_seq', dataset_name=dataset_name,
                            run_ids=None, display_name='PU-ARTrack_seq'))

dataset = get_dataset('hut290')

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

dataset_name = 'uot'
dataset = get_dataset('uot')

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
dataset_name = 'utb'
dataset = get_dataset('utb')

print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

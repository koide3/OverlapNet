#!/usr/bin/env python3
import yaml
import sys, os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/two_heads'))
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from infer import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/utils'))
from utils import *
import gen_depth_data as gen_depth
import gen_normal_data as gen_normal
import gen_intensity_data as gen_intensity
import gen_semantic_data as gen_semantics


def gen_data(seq_folder):
	scan_folder = '%s/velodyne' % seq_folder
	semantic_folder = '%s/semantic_probs' % seq_folder
	dst_folder = '%s/preprocess_data' % seq_folder

	range_data = gen_depth.gen_depth_data(scan_folder, dst_folder)[0]
	normal_data = gen_normal.gen_normal_data(scan_folder, dst_folder)[0]
	intensity_data = gen_intensity.gen_intensity_data(scan_folder, dst_folder)[0]

	os.makedirs(semantic_folder, exist_ok=True)


def extract_features(seq_folder):
	network_config_filename = 'config/network.yml'
	network_config = yaml.load(open(network_config_filename))
	network_config['infer_seqs'] = '/datasets/kitti/sequences/00/preprocess_data'

	scan_folder = seq_folder + '/velodyne'
	filenames = [scan_folder + '/' + x for x in os.listdir(scan_folder) if '.bin' in x]
	filenames = [os.path.basename(x).replace('.bin', '') for x in filenames]

	infer = Infer(network_config)
	infer.filenames = np.array(filenames)
	print(infer.datasetpath, infer.seq)

	feature_volumes = infer.create_feature_volumes(infer.filenames)
	print(feature_volumes.shape)

	features = np.float32(feature_volumes)
	features_folder = seq_folder + '/features'
	os.makedirs(features_folder, exist_ok=True)
	for feature, filename in zip(features, filenames):
		print(filename, feature.shape)
		np.save(features_folder + '/' + filename, feature)


if __name__ == '__main__':
	for i in range(0, 6):
		seq_folder = '/datasets/kitti/sequences/%02d' % i
		os.makedirs(seq_folder + '/preprocess_data', exist_ok=True)
		gen_data(seq_folder)
		extract_features(seq_folder)

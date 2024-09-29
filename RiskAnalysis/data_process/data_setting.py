import os
class data_setting():
	def __init__(self, input_paths, output_paths, class_num, base_risk_num, dtype='dis'):
		self.input_paths = input_paths
		self.output_paths = output_paths
		self.class_num = class_num
		self.base_risk_num = base_risk_num
		self.dtype = dtype

		self.train_all_sim = os.path.join(input_paths, 'train_all_sim.npy')
		self.train_all_dis = os.path.join(input_paths, 'train_all_dis.npy')
		self.train_file_paths = os.path.join(input_paths, 'train_file_paths.npy')
		self.train_labels = os.path.join(input_paths, 'train_labels.npy')

		self.val_all_sim = os.path.join(input_paths, 'val_all_sim.npy')
		self.val_all_dis = os.path.join(input_paths, 'val_all_dis.npy')
		self.val_file_paths = os.path.join(input_paths, 'val_file_paths.npy')
		self.val_labels = os.path.join(input_paths, 'val_labels.npy')

		self.test_all_sim = os.path.join(input_paths, 'test_all_sim.npy')
		self.test_all_dis = os.path.join(input_paths, 'test_all_dis.npy')
		self.test_file_paths = os.path.join(input_paths, 'test_file_paths.npy')
		self.test_labels = os.path.join(input_paths, 'test_labels.npy')

		self.write_all_data_info = 'all_data_info.csv'

		self.write_train = 'train.csv'
		self.write_val = 'val.csv'
		self.write_test = 'test.csv'
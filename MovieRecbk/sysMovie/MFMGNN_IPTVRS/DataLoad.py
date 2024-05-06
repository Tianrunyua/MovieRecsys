import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
class MyDataset(Dataset):
	def __init__(self, path, num_user, num_item):
		super(MyDataset, self).__init__()
		self.data = np.load(path+'/train.npy').T #iptv
		#self.data = np.load(path+'/train.npy') #非IPTV
		self.adj_lists = np.load(path + '/user_item_dict_all.npy', allow_pickle=True).item()
		self.num_user = num_user
		self.num_item = num_item
		self.all_set = set(range(num_user, num_user+num_item))

	def __getitem__(self, index):
		user, pos_item = self.data[index]
		while True:
			neg_item = np.random.randint(self.num_user, self.num_user + self.num_item)
			# pdb.set_trace()
			if neg_item not in self.adj_lists[user]:
				break

		return [int(user), int(pos_item), int(neg_item)]
		# return self.data
	def __len__(self):
		return len(self.data)



if __name__ == '__main__':
	num_item = 100
	num_user = 36656
	dataset = MyDataset('../../Movielens/', num_user, num_item)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

	for data in dataloader:
		user, pos_items, neg_items= data



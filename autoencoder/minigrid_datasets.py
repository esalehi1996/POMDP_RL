import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image


# ------------------------------------------------ 3 x 3 Observation grids ------------------------------------------------




# ------------------------------------------------ 7 x 7 Observation grids ------------------------------------------------

#Parent Class for 7x7 Observation Grids
class ObsGrids7x7(data.Dataset):
	def __init__(self, data_folder):
		self.data_folder = data_folder
		self.training_file = 'training.pt'
		self.data = torch.load(os.path.join(self.data_folder, self.training_file))
		self.mean = torch.load(os.path.join(self.data_folder, 'mean.pt'))
		self.max_vals = torch.load(os.path.join(self.data_folder, 'max_vals.pt'))*1.2
		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.max_vals)])

	def __getitem__(self, index):
		data_item = self.data[index]
		data_item = Image.fromarray(data_item)
		if self.transform is not None:
			data_item = self.transform(data_item)
		return data_item.reshape(-1)

	def __len__(self):
		return len(self.data)


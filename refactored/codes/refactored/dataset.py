import torch
import numpy as np
import os
class RotMNIST(torch.utils.data.Dataset):
	'''
	Class which returns (image,label,angle,bin)
	TODO - 1. Shuffle indices, bin and append angles
		   2. Return 
		   3. OT - make sure OT takes into account the labels, i.e. OT loss should be inf for interchanging labels.  
	'''
	def __init__(self,indices,transported_samples=None,target_bin=None,**kwargs):
		'''
		'''
		self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
		self.transported_samples = transported_samples  # a 2-d array of OT maps
		root = kwargs['data_path']
		self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
		self.target_bin = target_bin
		self.X = np.load("{}/X.npy".format(root))
		self.Y = np.load("{}/Y.npy".format(root))
		self.A = np.load("{}/A.npy".format(root))
		self.U = np.load("{}/U.npy".format(root))
		self.device = kwargs['device']
	def __getitem__(self,idx):
		index = self.indices[idx]
		X = torch.tensor(self.X[idx]).float().to(self.device)   # Check if we need to reshape
		Y = torch.tensor(self.Y[idx]).long().to(self.device)
		A = torch.tensor(self.A[idx]).float().to(self.device).view(1)
		U = torch.tensor(self.U[idx]).float().to(self.device).view(1)
		if self.transported_samples is not None:
			source_bin = int(np.round(U.item() * self.num_bins)) 
			transported_X = torch.from_numpy(self.transported_samples[source_bin][self.target_bin][idx % 1000]).float().to(self.device) #This should be similar to index fun, an indexing function which takes the index of the source sample and returns the corresponding index of the target sample.
			return X,transported_X,A,U,Y

		return X,A,U,Y

	def __len__(self):
		return len(self.indices)

class ClassificationDataSet(torch.utils.data.Dataset):
	
	def __init__(self, data_path, indices, target_bin=None, **kwargs):
		'''
		TODO: Handle OT
		Pass Transported samples as kwargs?
		'''
		self.indices = indices # Indices are the indices of the elements from the arrays which are emitted by this data-loader
		self.transported_samples = None  # a 2-d array of OT maps
		
		self.root = data_path
		self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
		#self.num_bins = kwargs['num_bins']  # using this we can get the bin corresponding to a U value
		
		self.target_bin = target_bin
		self.X = np.load("{}/X.npy".format(self.root))
		self.Y = np.load("{}/Y.npy".format(self.root))
		self.A = np.load("{}/A.npy".format(self.root))
		self.U = np.load("{}/U.npy".format(self.root))
		
	def __getitem__(self,idx):

		index = self.indices[idx]
		data = torch.tensor(self.X[idx]).float().to(self.device)   # Check if we need to reshape
		label = torch.tensor(self.Y[idx]).long().to(self.device)
		auxiliary = torch.tensor(self.A[idx]).float().to(self.device).view(-1, 1)
		domain = torch.tensor(self.U[idx]).float().to(self.device).view(-1, 1)
		if self.transported_samples is not None:
			source_bin = int(np.round(U.item() * self.num_bins)) 
			transported_X = torch.from_numpy(self.transported_samples[source_bin][self.target_bin][idx % 1000]).float().to(self.device) #This should be similar to index fun, an indexing function which takes the index of the source sample and returns the corresponding index of the target sample.
			return X,transported_data, auxiliary, domain,  label

		return data, auxiliary, domain, label

	def __len__(self):
		return len(self.indices)


class GradDataset(torch.utils.data.Dataset):
	'''
	Class which returns (image,label,angle,bin)
	TODO - 1. Shuffle indices, bin and append angles: Isn't shuffling done by DataLoader that wraps this dataset instance?
		   2. Return 
		   3. OT - make sure OT takes into account the labels, i.e. OT loss should be inf for interchanging labels.  
	'''

	def __init__(self, data_path, src_indices, target_indices, target_bin=None, n_samples=6000, **kwargs):
	
		'''
		You give it a set of indices, along with which bins they belong
		It returns images from that MNIST bin
		usage - indices = np.random.shuffle(np.arange(n_samples)) 
		'''

		self.root = data_path
		
		self.src_indices = src_indices # np.random.shuffle(np.arange(n_samples))
		self.target_indices = target_indices # np.random.shuffle(np.arange(n_samples))
		self.target_labs = {}
		
		self.X = np.load("{}/X.npy".format(self.root))
		self.Y = np.load("{}/Y.npy".format(self.root))
		self.A = np.load("{}/A.npy".format(self.root))
		self.U = np.load("{}/U.npy".format(self.root))
		
		self.device = kwargs['device'] if kwargs.get('device') else 'cpu'
		
		self.target_bin = target_bin

		# print(self.bins,self.bin_width)
		# print("---------- READING MNIST ----------")

		#for i in self.target_indices:
		#	if self.Y[i].item() not in self.target_labs.keys():
		#		self.target_labs[self.Y[i].item()] = [i]
		#	else:
		#		self.target_labs[self.Y[i].item()].append(i)

	def __getitem__(self, idx):
		
		index = self.src_indices[idx]
		
		data = torch.tensor(self.X[index])
		label = torch.tensor(self.Y[index])
		a_info = torch.tensor(self.A[index])
		

		#target_ids = self.target_labs[label.item()]
		#target_idx = target_ids[idx % len(target_ids)]

		target_idx = self.target_indices
		target_data = torch.tensor(self.X[target_idx])
		a_info_target = torch.tensor(self.A[target_idx])

		# label = self.Y[index]
		
		# print(bin,norm_angle)

		return data.to(self.device),target_data.to(self.device), (a_info-a_info_target).float().to(self.device)#, U[index].float().to(self.device), label.long().to(self.device)
	
	def __len__(self):
		return len(self.src_indices)        
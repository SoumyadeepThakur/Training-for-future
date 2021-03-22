import pandas as pd
import numpy as np
import math
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
import os
from PIL import Image
from torchvision.transforms.functional import rotate
import matplotlib.pyplot as plt

def load_sleep(filename):

	domains = 5
	
	df =  pd.read_csv(filename)
	df = df.drop(['rcrdtime'], axis=1)
	nan_values = dict()
	for col in df.columns:
		nan_values[col] = df[col].isna().sum()

	df = df.dropna(subset=['Staging1','Staging2','Staging3','Staging4','Staging5'])
	final_cols = []
	for col in nan_values.keys():
		if nan_values[col] <= 500:
			final_cols.append(col)

	print(len(final_cols))
	df = df[final_cols]
	imputer = SimpleImputer(strategy='mean')
	#imputer = KNNImputer(n_neighbors=3, weights="uniform")
	df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
	print(df.shape)


	X_data, Y_data, A_data, U_data = [], [], [], []

	ckpts = [50, 60, 70, 80, 90]

	for i, ckpt in enumerate(ckpts):

		data_temp = df[df['age_s1'] <= ckpt]
		df = df[df['age_s1'] > ckpt]
		Y_temp = data_temp['Staging1'].values
		Y_temp = np.eye(2)[Y_temp.astype(np.int32)]
		#A_temp = (data_temp['age_s1'].values-39)/90
		A_temp = (data_temp['age_s1'].values-38)/90
		data_temp = data_temp.drop(['Staging1'], axis=1)
		X_temp = data_temp.drop(['Staging2', 'Staging3', 'Staging4', 'Staging5', 'age_s1', 'age_category_s1'], axis=1).values
		#U_temp = np.array([i]*X_temp.shape[0])*1.0/5
		U_temp = np.array([i+1]*X_temp.shape[0])*1.0/5
		print(X_temp.shape)
		print(Y_temp.shape)
		print(A_temp.shape)
		print(U_temp.shape)

		X_temp =X_temp.astype(np.float32)
		A_temp = A_temp.astype(np.float32)
		U_temp =U_temp.astype(np.float32)
		
		X_data.append(X_temp)
		Y_data.append(Y_temp)
		A_data.append(A_temp)
		U_data.append(U_temp)

	return np.array(X_data), np.array(Y_data), np.array(A_data), np.array(U_data)

def preprocess_sleep(X_data, base_domains):

	X_all = np.vstack(X_data[base_domains])
	scaler = StandardScaler(copy=False)
	#scaler = MinMaxScaler(copy=False)
	scaler.fit(X_all)

	for i in range(len(X_data)):
		scaler.transform(X_data[i])

	return X_data

def load_moons(domains):

	X_data, Y_data, U_data = [], [], []
	for i in range(domains):

		angle = i*math.pi/(domains-1);
		X, Y = make_moons(n_samples=200, noise=0.1, random_state=2701)
		rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
		X = np.matmul(X, rot)
		
		#plt.scatter(X[:,0], X[:,1], c=Y)
		#plt.savefig('moon_%d' % i)
		#plt.clf()

		Y = np.eye(2)[Y]
		U = np.array([i*1.0] * 200)/domains

		X_data.append(X)
		Y_data.append(Y)
		U_data.append(U)

	return np.array(X_data), np.array(Y_data), np.array(U_data)

def load_Rot_MNIST(use_vgg, root="./MNIST_data"):

	mnist_ind = (np.arange(60000))
	np.random.seed(2701)
	np.random.shuffle(mnist_ind)
	mnist_ind = mnist_ind[:6000]
	# Save indices
	processed_folder = os.path.join(root, 'MNIST', 'processed')
	data_file = 'training.pt'
	vgg_means = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
	vgg_stds  = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
	data, targets = torch.load(os.path.join(processed_folder, data_file))
	X_data = []
	Y_data = []
	A_data = []
	U_data = []
	X = []
	Y = []
	U = []
	A = []
	all_indices = [[x for x in range(i*1000,(i+1)*1000)] for i in range(6)]

	
	for idx in range(len(mnist_ind)):
		index = mnist_ind[idx]
		if idx%1000 == 0 and idx > 0:
			X_data.append(np.stack(X))
			Y_data.append(np.vstack(Y))
			U_data.append(np.hstack(U))
			A_data.append(np.hstack(A))
			X = []
			Y = []
			U = []
			A = []	

		bin = int(idx / 1000)
		angle = bin * 15
		image = data[index]
		image = Image.fromarray(image.numpy(), mode='L')
		image = np.array(rotate(image,angle))#).float().to(device)
		image = image / 255.0
		if use_vgg:
			image = image.reshape((1,28,28)).repeat(3,axis=0)
			image = (image - vgg_means)/vgg_stds
		else:
			image = image.reshape((1,28,28))
			# image = (image - vgg_means)/vgg_stds

		#plt.imshow(image[0])
		#plt.show()
		I_y = np.eye(10)
		X.append(image)
		Y.append(I_y[targets[index]])
		U.append(bin/6)
		A.append(angle/90)

	X_data.append(np.stack(X))
	Y_data.append(np.vstack(Y))
	U_data.append(np.hstack(U))
	A_data.append(np.hstack(A))
	X = []
	Y = []
	U = []
	A = []
		
	return np.array(X_data), np.array(Y_data), np.array(A_data), np.array(U_data)
	#np.save("{}/X.npy".format(processed_folder),np.stack(all_images),allow_pickle=True)
	#np.save("{}/Y.npy".format(processed_folder),np.array(all_labels),allow_pickle=True)
	#np.save("{}/A.npy".format(processed_folder),np.array(all_A),allow_pickle=True)
	#np.save("{}/U.npy".format(processed_folder),np.array(all_U),allow_pickle=True)
	#json.dump( all_indices,open("{}/indices.json".format(processed_folder),"w"))
	# json.dump(all_indices, open("{}/indices.json".format(processed_folder),"w"))

def load_housing(filename):
	
	X_data, Y_data, U_data, A_data = [], [], [], []

	post_dict = {2607: 1, 2906: 2, 2905: 3, 2606: 4, 2902: 5, 2612: 6, 2904: 7, 2615: 8, 2914: 9, 2602: 10, 2600: 11,
       2605: 12, 2603: 13, 2611: 14, 2903: 15, 2617: 16, 2913: 17, 2604: 18, 2614: 19, 2912: 20, 2601: 21, 2900: 22,
       2620: 23, 2618: 24, 2616: 25, 2911: 26, 2609: 27}


	df = pd.read_csv(filename)
	df = df[df['propertyType'] == 'house']
	df['daysepoch'] = df['datesold'].apply(lambda d: (np.datetime64(d.split()[0]) - np.datetime64('2007-01-01')).astype(int))
	maxdays = df['daysepoch'].max()
	years = list(range(2008, 2021))
	k=0
	for y in years:

		y_str = "%d-01-01 00:00:00" %y
		dom_df = df[df['datesold'] < y_str]
		df = df[df['datesold'] >= y_str]
		#dom_df['datesold'] = dom_df['datesold'].apply(lambda d: (np.datetime64(d.split()[0]) - np.datetime64('%d-01-01' %(y-1))).astype(int))
		dom_df.to_csv("sales_%d.csv" %k, index=False)
		k+=1
		Y_data.append(dom_df['price'].values.reshape(-1,1)/10000.0)
		posts = np.eye(28)[np.array([post_dict[x] for x in dom_df['postcode'].values])]
		X_data.append(np.hstack([posts, dom_df['bedrooms'].values.reshape(-1,1)]))
		U_data.append(np.array([y-2007]*dom_df.shape[0])/13.0)
		A_data.append(dom_df['daysepoch'].values/365)

	return np.array(X_data), np.array(Y_data), np.array(A_data), np.array(U_data)


def load_m5(trainfile, testfile):

	X_data, Y_data, A_data, U_data = [], [], [], []

	sc = StandardScaler()
	train = pd.read_csv(trainfile)

	ckpts = ['2015-01-01', '2016-01-01']

	for i, ckpt in enumerate(ckpts):

		print('Dom %d' %i)

		cur = train[train['date'] < ckpt]
		train = train[train['date'] >= ckpt]
		cur = cur.drop(['date', 'part', 'id'], axis=1)

		Y = cur['demand'].values.astype(np.float32)
		X = cur.drop(['demand'], axis=1).values.astype(np.float32)
		if i == 0:
			X = sc.fit_transform(X)
		else:
			X = sc.transform(X)
			
		U = np.array([i]*X.shape[0])
		A = np.array([i]*X.shape[0]) + cur['month'].values/12.0

		X_data.append(X)
		Y_data.append(Y)
		U_data.append(U)
		A_data.append(A)

	test = pd.read_csv(testfile)

	Y = cur['demand'].values.astype(np.float32)
	X = cur.drop(['demand'], axis=1).values.astype(np.float32)
	U = np.array([len(ckpts)]*X.shape[0])
	A = np.array([len(ckpts)]*X.shape[0]) + cur['month'].values/12.0

	X_data.append(X)
	Y_data.append(Y)
	U_data.append(U)
	A_data.append(A)	

	return np.array(X_data), np.array(Y_data), np.array(A_data), np.array(U_data)


import pandas as pd
import numpy as np
import math
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer

def load_sleep2(filename):

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
		A_temp = (data_temp['age_s1'].values-39)/90
		data_temp = data_temp.drop(['Staging1'], axis=1)
		X_temp = data_temp.drop(['Staging2', 'Staging3', 'Staging4', 'Staging5', 'age_s1', 'age_category_s1'], axis=1).values
		U_temp = np.array([i]*X_temp.shape[0])

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

def preprocess_sleep2(X_data, base_domains):

	X_all = np.vstack(X_data[base_domains])
	scaler = StandardScaler(copy=False)
	#scaler = MinMaxScaler(copy=False)
	scaler.fit(X_all)

	for i in range(len(X_data)):
		scaler.transform(X_data[i])

	return X_data

def load_sleep(filename):

	domains = 5
	
	df =  pd.read_csv(filename)
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
	df = df.fillna(0)
	print(df.shape)


	X_data, Y_data, U_data = [], [], []

	ckpts = [50, 60, 70, 80, 90]

	for i, ckpt in enumerate(ckpts):

		data_temp = df[df['age_s1'] <= ckpt]
		df = df[df['age_s1'] > ckpt]
		Y_temp = data_temp['Staging1'].values
		Y_temp = np.eye(2)[Y_temp.astype(np.int32)]
		data_temp = data_temp.drop(['Staging1', 'rcrdtime'], axis=1)
		X_temp = data_temp.drop(['Staging2', 'Staging3', 'Staging4', 'Staging5', 'age_s1', 'age_category_s1'], axis=1).values
		U_temp = np.array([i]*X_temp.shape[0])

		print(X_temp.shape)
		print(Y_temp.shape)
		print(U_temp.shape)
		X_temp =X_temp.astype(np.float32)
		U_temp =U_temp.astype(np.float32)
		X_data.append(X_temp)
		Y_data.append(Y_temp)
		U_data.append(U_temp)

	return np.array(X_data), np.array(Y_data), np.array(U_data)

def preprocess_sleep(X_data, Y_data, U_data, base_domains):

	X_all = np.vstack(X_data[base_domains])
	scaler = StandardScaler(copy=False)
	scaler.fit(X_all)

	for i in range(len(X_data)):
		scaler.transform(X_data[i])

	return X_data, Y_data, U_data

def load_moons(domains):

	X_data, Y_data, U_data = [], [], []
	for i in range(domains):

		angle = i*math.pi/(domains-1);
		X, Y = make_moons(n_samples=200, noise=0.1)
		rot = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
		X = np.matmul(X, rot)
		
		#plt.scatter(X[:,0], X[:,1], c=Y)
		#plt.savefig('moon_%d' % i)
		#plt.clf()

		Y = np.eye(2)[Y]
		U = np.array([i] * 200)

		X_data.append(X)
		Y_data.append(Y)
		U_data.append(U)

	return np.array(X_data), np.array(Y_data), np.array(U_data)

if __name__=="__main__":

	X_data, Y_data, U_data = load_sleep('shhs1-dataset-0.15.0.csv')


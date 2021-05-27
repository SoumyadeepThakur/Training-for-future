from losses import *



class Config():
	def __init__(self,args):
		self.epoch_classifier = args.epoch_classifier
		self.epoch_finetune = args.epoch_finetune 
		self.SUBEPOCHS = 1
		self.EPOCH = args.epoch_finetune // self.SUBEPOCHS
		self.bs = args.bs
		self.CLASSIFICATION_BATCH_SIZE = 100
		# self.PRETRAIN_EPOCH = 5
		self.data = args.data 
		self.update_num_steps = 1
		self.num_finetune_domains = 2
		self.delta = args.delta
		self.max_k = args.max_k
		self.w_decay = 0
		self.schedule = False

		
		log_file_name = 'debugging_no_curric_{}_{}'.format(args.data,args.model)
		'''
		if args.trelu_limit < 1000:
			log_file_name += '_trelulimit-{}'.format(args.trelu_limit)
		if args.single_trelu:
			log_file_name += '_single-trelu'
		if args.delta > 0.0 or args.delta < 0.0:
			log_file_name += '_delta-{}'.format(args.delta)
		if args.max_k > 1:
			log_file_name += '_max-k-{}'.format(args.max_k)			
		if args.time_softmax:
			log_file_name += '_time-softmax'
		if args.ensemble:
			log_file_name += '_ensemble'
		'''
		self.log = open(log_file_name,"a")
		

		if args.data == "house":

			self.dataset_kwargs = {"root_dir":"../../data/HousePrice","device":args.device, "drop_cols":None}
			self.source_domain_indices = [6,7,8,9,10]
			self.target_domain_indices = [11]
			self.data_index_file = "../../data/HousePrice/indices.json"
			from models_GI import ClassifyNetHuge
			self.classifier = ClassifyNetHuge 
			self.model_kwargs =  {'time_conditioning':True,'task':'regression','use_time2vec':True,'leaky':True,"input_shape":31,"hidden_shapes":[400,400,400],"output_shape":1,'append_time':True}
			self.lr = 5e-4
			self.lr_reduce = 1.0
			self.classifier_loss_fn = reconstruction_loss
			self.loss_type = 'regression'
			self.encoder = None

			self.delta_lr=0.1
			self.delta_clamp=0.15
			self.delta_steps=10
			self.lambda_GI=0.5

		if args.data == "mnist":

			self.dataset_kwargs = {"root_dir":"../../data/MNIST/processed/","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0,1,2,3]
			self.target_domain_indices = [4]
			self.data_index_file = "../../data/MNIST/processed/indices.json"
			from models_GI import ResNet, ResidualBlock
			self.classifier = ResNet 
			self.model_kwargs =  {
									"block": ResidualBlock,
									"layers": [2, 2, 2, 2],
									"time_conditioning": True,
									"leaky": True,
									"append_time": True,
									"use_time2vec": True
								}
			self.lr = 1e-4
			self.classifier_loss_fn = classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.lr_reduce = 5.0
			self.delta_lr=0.05
			self.delta_clamp=0.05
			self.delta_steps=5
			self.lambda_GI=1.0

		if args.data == 'moons':

			self.dataset_kwargs = {"root_dir":"../../data/Moons/processed", "device":args.device, "drop_cols":None}
			self.source_domain_indices = [0,1, 2, 3, 4, 5, 6, 7, 8]
			self.target_domain_indices = [9]
			self.data_index_file = "../../data/Moons/processed/indices.json"
			from models_GI import PredictionModel
			self.classifier = PredictionModel
			self.model_kwargs =  {"input_shape":3, "hidden_shapes":[50, 50], "out_shape":1, "time_conditioning": True, "trelu": True, "use_time2vec":True, 
									"leaky":True, "regression": False}
			self.lr = 5e-3
			self.classifier_loss_fn = binary_classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.delta_lr=0.05
			self.delta_clamp=0.5
			self.delta_steps=5
			self.lambda_GI=1.0e-4
			self.lr_reduce=10
			self.multistep = args.multistep

		if args.data == 'onp':

			self.dataset_kwargs = {"root_dir":"../../data/ONP/processed","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0, 1, 2, 3, 4]
			self.target_domain_indices = [5]
			self.data_index_file = "../../data/ONP/processed/indices.json"
			from models_GI import PredictionModel
			self.classifier = PredictionModel
			self.model_kwargs =  {"input_shape":59, "hidden_shapes":[200], "out_shape":1, "time_conditioning": True, "trelu": True, "use_time2vec":False, 
									"leaky":True, "regression": False}
			#self.model_kwargs = {"data_shape": 59, "hidden_shapes": [200], "out_shape": 1, "time_conditioning": True, "trelu": False, "time2vec": False}
			self.lr = 1e-3
			self.classifier_loss_fn = binary_classification_loss
			self.loss_type = 'classification'
			self.encoder = None

			self.delta_lr=0.5
			self.delta_clamp=0.1
			self.delta_steps=10
			self.lambda_GI=1.0
			self.lr_reduce=10.0
			self.w_decay = 1e-4


		if args.data == 'm5':

			

			self.dataset_kwargs = {"root_dir":"../../data/M5/processed","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0, 1, 2]
			self.target_domain_indices = [3]
			self.data_index_file = "../../data/M5/processed/indices.json"
			#from models_GI import PredictionModel
			from models_GI import M5Model
			self.classifier = M5Model
			#self.model_kwargs =  {"input_shape":75, "hidden_shapes":[50, 50], "out_shape":1, "time_conditioning": True, "use_time2vec":True,
			#						"leaky":True, "regression": True}
			self.model_kwargs = {"data_shape": 75, "hidden_shape": 50, "out_shape": 1, "time_conditioning": True, "trelu": True, "time2vec": True}
			self.lr = 1e-2
			self.classifier_loss_fn = reconstruction_loss
			self.loss_type = 'regression'
			self.encoder = None

			self.delta_steps=5
			self.w_decay = 1e-4
			self.delta_lr=0.5
			self.delta_clamp=0.5
			self.delta_steps=5
			self.lambda_GI=1.0
			self.lr_reduce=20.0
			self.schedule = True


		if args.data == 'm5_household':

			

			self.dataset_kwargs = {"root_dir":"../../data/M5/processed_household","device":args.device, "drop_cols":None}
			self.source_domain_indices = [0, 1, 2]
			self.target_domain_indices = [3]
			self.data_index_file = "../../data/M5/processed_household/indices.json"
			#from models_GI import PredictionModel
			from models_GI import M5Model
			self.classifier = M5Model
			#self.model_kwargs =  {"input_shape":75, "hidden_shapes":[50, 50], "out_shape":1, "time_conditioning": True, "use_time2vec":True,
			#						"leaky":True, "regression": True}
			self.model_kwargs = {"data_shape": 75, "hidden_shape": 50, "out_shape": 1, "time_conditioning": True, "trelu": True, "time2vec": True}
			self.lr = 1e-2
			self.classifier_loss_fn = reconstruction_loss
			self.loss_type = 'regression'
			self.encoder = None

			self.delta_steps=5
			self.w_decay = 1e-4
			self.delta_lr=0.5
			self.delta_clamp=0.5
			self.delta_steps=5
			self.lambda_GI=1.0
			self.lr_reduce=20.0
			self.schedule = True



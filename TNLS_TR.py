import math
import numpy as np, os, sys, re, glob, subprocess, math, unittest, shutil, time, string, logging, gc
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
np.set_printoptions(precision=4)
from time import gmtime, strftime
from random import shuffle, choice, sample, choices
import random
from itertools import product
from functools import partial
import inspect
import itertools



base_folder = './'
try:
	os.mkdir(base_folder+'log')
	os.mkdir(base_folder+'agent_pool')
	os.mkdir(base_folder+'job_pool')
	os.mkdir(base_folder+'result_pool')
except:
	pass


current_time = strftime("%Y%m%d_%H%M%S", gmtime())

log_name = 'sim_{}_{}.log'.format('data', sys.argv[2])
logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
										format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class DummyIndv(object): pass

data = np.load('data.npz')
logging.info(data['adj_matrix'])
adjm=data['adj_matrix']
adjm[adjm==0] = 1
actual_elem = np.sum([ np.prod(adjm[d]) for d in range(adjm.shape[0]) ])
logging.info(actual_elem)
np.save('data.npy', data['goal'])
evoluation_goal = 'data.npy'

class Individual(object):
	def __init__(self, adj_matrix=None, scope=None, **kwargs):
		super(Individual, self).__init__()
		if adj_matrix is None:
			self.adj_matrix = kwargs['adj_func'](**kwargs)
		else:
			self.adj_matrix = adj_matrix
		self.scope = scope
		self.dim = self.adj_matrix.shape[0]
		adj_matrix_T=np.copy(self.adj_matrix)
		size_data=np.arange(self.dim)
		for i in range(self.dim):
				size_data[i]=adj_matrix_T[i][0][0][0]
		
		rank=np.arange(self.dim)
		for i in range(self.dim):
				rank[i]=adj_matrix_T[i][0][0][1]

		permute_code=np.arange(0, self.dim)
		for i in range(self.dim):
				permute_code[i]=adj_matrix_T[i][0][1]
		adj_matrix_R = np.diag(size_data)
		temp=np.arange(self.dim-1)
		temp=temp[::-1]
		temp[0]=self.dim-3
		connection_index = []
		connection_index.append(0)
		for i in range(self.dim):
				if i==1:
						connection_index.append(connection_index[-1]+1)
				else:
						if i==0:
								connection_index.append(connection_index[-1]+(temp[i]+1))
						else:
								connection_index.append(connection_index[-1]+(temp[i-1]+1))
		connection_index=connection_index[0:self.dim]
		connection =rank.tolist()
		index_tuple=np.triu_indices(self.dim, 1)
		index_tuple1=index_tuple[0]
		index_tuple2=index_tuple[1]
		index_tuple1=index_tuple1[connection_index]
		index_tuple2=index_tuple2[connection_index]
		index_tuple=[index_tuple1,index_tuple2]
		index_tuple=tuple(index_tuple)
		adj_matrix_R[index_tuple] = connection
		adj_matrix_R[np.tril_indices(self.dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(self.dim, -1)]
		index=np.diag_indices_from(adj_matrix_R)
		adj_matrix_R[index]=0

		permute=permute_code

		permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
		for i in range(self.dim):
				permutation_matrix[permute[i],i] = 1

		adj_matrix_R=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())




		self.parents = kwargs['parents'] if 'parents' in kwargs.keys() else None
		self.repeat = kwargs['evaluate_repeat'] if 'evaluate_repeat' in kwargs.keys() else 1
		self.iters = kwargs['max_iterations'] if 'max_iterations' in kwargs.keys() else 10000
		self.rse_therhold=kwargs['rse_therhold'][0]

		adj_matrix_k = np.copy(adj_matrix_R)
		adj_matrix_k[adj_matrix_k==0] = 1

		self.present_elements = np.prod(np.diag(adj_matrix_k))
		self.actual_elements = np.sum([ np.prod(adj_matrix_k[d]) for d in range(self.dim) ])
		self.sparsity = self.actual_elements/self.present_elements

	def deploy(self, sge_job_id):
		try:
			path = base_folder+'/job_pool/{}.npz'.format(sge_job_id)
			np.savez(path, adj_matrix=self.adj_matrix, scope=self.scope, repeat=self.repeat, iters=self.iters,rse_TH=self.rse_therhold)
			self.sge_job_id = sge_job_id
			return True
		except Exception as e:
			raise e

	def collect(self, fake_loss=False):
		if not fake_loss:
			try:
				path = base_folder+'/result_pool/{}.npz'.format(self.scope.replace('/', '_'))
				result = np.load(path)
				self.repeat_loss = result['repeat_loss']

				os.remove(path)
				return True
			except Exception:
				return False
		else:
			self.repeat_loss = [9999]*self.repeat
			return True

class Generation(object):
	def __init__(self, CG,RC,FC,n_generation,pG=None, name=None, **kwargs):
		super(Generation, self).__init__()
		self.name = name
		self.N_islands = kwargs['N_islands'] if 'N_islands' in kwargs.keys() else 1
		self.kwargs = kwargs
		self.out = self.kwargs['out']
		self.rank = self.kwargs['rank']
		self.size = self.kwargs['size']
		self.indv_to_collect = []
		self.n_generation = n_generation 



		self.center_generation=CG
		self.rank_center=RC
		self.fitness_score_center=FC

		self.indv_to_distribute = []
		if pG is not None:
			self.societies = {}
			for k, v in pG.societies.items():
				self.societies[k] = {}
				self.societies[k]['indv'] = \
						[ Individual( adj_matrix=indv.adj_matrix, parents=indv.parents,
													scope='{}/{}/{:03d}'.format(self.name, k, idx), **self.kwargs) \
						for idx, indv in enumerate(v['indv']) ]
				self.indv_to_distribute += [indv for indv in self.societies[k]['indv']]

		elif 'random_init' in kwargs.keys():
			self.societies = {}
			for n in range(self.kwargs['N_islands']):
				society_name = ''.join(choice(string.ascii_uppercase + string.digits) for _ in range(6))
				self.societies[society_name] = {}
				self.societies[society_name]['indv'] = [ \
						Individual(scope='{}/{}/{:03d}'.format(self.name, society_name, i), 
						adj_func=self.__random_adj_matrix__, **self.kwargs) \
						for i in range(1) ]
				self.indv_to_distribute += [indv for indv in self.societies[society_name]['indv']]

	def __call__(self, **kwargs):
		try:
			self.__evaluate__()
			if 'callbacks' in kwargs.keys():
				for c in kwargs['callbacks']:
					c(self)
			self.__evolve__()
			return True
		except Exception as e:
			raise e

	def __random_adj_matrix__(self, **kwargs):
		if isinstance(self.out, list):
			adj_matrix = np.diag(self.out)
		else:
			adj_matrix = np.diag(self.out)
		out=self.out
		rank=np.random.randint(1, self.rank, (self.size,1))

		permute=np.arange(1,self.size)
		np.random.shuffle(permute)
		permute=np.insert(np.array([0],dtype=int),1,permute,0)
		np.random.shuffle(permute)

		permute=permute.reshape((self.size,1))
		out=out.reshape((self.size,1))
		adj_matrix=np.rec.fromarrays((out, rank))
		adj_matrix=np.rec.fromarrays((adj_matrix, permute))
		return adj_matrix

	def __evolve__(self):

		def TN_structure_LocalSampling(island,generation):

			if generation==1:
				self.center_generation=generation
				temp=np.copy(island['indv'][0].adj_matrix)
				rank=np.arange(self.size)
				for i in range(self.size):
								rank[i]=temp[i][0][0][1]

				permute_code=np.arange(0,self.size)
				for i in range(self.size): 
								permute_code[i]=temp[i][0][1]

				self.rank_center=island['indv'][0].adj_matrix
				self.fitness_score_center=island['total'][0]



				Rvarien=self.kwargs['c1'][0]**(generation-1)*self.kwargs['R_varience_init'][0]
				if Rvarien<self.kwargs['R_varience_LB'][0]:
					Rvarien=self.kwargs['R_varience_LB'][0]
                    
				Pvarien=self.kwargs['c2'][0]**(generation-1)*self.kwargs['P_varience_init'][0]
				if Pvarien<self.kwargs['P_varience_LB'][0]:
					Pvarien=self.kwargs['P_varience_LB'][0]                
                    

				num_rank=self.size
				adjmat_set=[]

				slot_R=np.arange(0,((self.kwargs['slot_R'][0]-1)-1)+1+1,1).astype(float)
				slot_R[slot_R>0]=slot_R[slot_R>0]-0.5

				slot_P=np.arange(0,((self.kwargs['slot_P'][0]-1)-1)+1+1,1).astype(float)
				#slot_P[slot_P>0]=slot_P[slot_P>0]-0.5

				exchange_set=[]
                
				rank_UB=np.where(rank==self.kwargs['rank']-1)
				rank_LB=np.where(rank==1)



				for samplen in range(self.kwargs['Sample_Numbers'][0]):
							temp=np.random.normal(0,Rvarien,[1,num_rank])
							positive_index=temp>=0
							temp_abs=np.abs(temp)

							for i in range(num_rank):
								rtemp=temp_abs[0,i]-slot_R
								rtemp=rtemp>0
								rtemp=rtemp.astype(int)
								rtemp=np.sum(rtemp)

								if rtemp>self.kwargs['slot_R'][0]-1:
									rtemp=self.kwargs['slot_R'][0]-1-1
								else:
									rtemp = rtemp - 1

								if positive_index[0,i]:
									temp[0,i]=rtemp
								else:
									temp[0,i]=-1*rtemp
							temp=temp.astype(int)
                            
                            
							UB=np.copy(temp[0,rank_UB])   
							LB=np.copy(temp[0,rank_LB])   
  							
							UB=-1*np.abs(UB)
							LB=np.abs(LB)
							temp[0,rank_UB]=UB
							temp[0,rank_LB]=LB
                            
							sampletemp=rank.reshape((1,num_rank))
							sampletemp=sampletemp+temp
							sampletemp[sampletemp>(self.rank-1)]=self.rank-1
							sampletemp[sampletemp<1]=1
							samplerank=sampletemp.reshape((num_rank,))

							if np.random.uniform() > Pvarien:
								out=self.out
								rank_temp=samplerank.reshape((self.size,1))
								permute_temp=permute_code.reshape((self.size,1))
								out=out.reshape((self.size,1))
								adj_matrix=np.rec.fromarrays((out, rank_temp))
								adj_matrix=np.rec.fromarrays((adj_matrix, permute_temp))
								adjmat_set.append(adj_matrix)
								exchange_set.append([0])
							else:
								temp=np.random.normal(0,self.kwargs['P_varience_R'][0],[1,1])
								positive_index=temp>=0
								temp_abs=np.abs(temp)



								for i in range(1):
									rtemp=temp_abs[0,i]-slot_P
									rtemp=rtemp>0
									rtemp=rtemp.astype(int)
									rtemp=np.sum(rtemp)

									if rtemp>self.kwargs['slot_P'][0]-1:
										rtemp=self.kwargs['slot_P'][0]-1
									else:
										rtemp = rtemp 
	
									if positive_index[0,i]:
										temp[0,i]=rtemp
									else:
										temp[0,i]=-1*rtemp
								temp=temp.astype(int)
								temp=np.abs(temp)
								samplepermute=np.copy(permute_code)
								ttset=[]
								for tempn in range(int(temp)):
														tnow=np.arange(0,self.size)
														np.random.shuffle(tnow)
														tnow=tnow[0:2]
														ttset=ttset+tnow.tolist()
														samplepermute[tnow]=samplepermute[tnow[::-1]]

								exchange_set.append(ttset)
								out=self.out
								rank_temp=samplerank.reshape((self.size,1))
								permute_temp=samplepermute.reshape((self.size,1))
								out=out.reshape((self.size,1))
								adj_matrix=np.rec.fromarrays((out, rank_temp))
								adj_matrix=np.rec.fromarrays((adj_matrix, permute_temp))
								adjmat_set.append(adj_matrix)



				for i in range(self.kwargs['Sample_Numbers'][0]-1):
								island['indv'].append(DummyIndv())
				for i in range(self.kwargs['Sample_Numbers'][0]):
								island['indv'][i].adj_matrix=adjmat_set[i]

								island['indv'][i].parents=('%d'%(self.center_generation)+'|'+''.join([str(k) for k in exchange_set[i]]))

			else:

				if island['total'][island['rank'][0]]<=self.fitness_score_center:
						self.fitness_score_center=island['total'][island['rank'][0]]
						temp=np.copy(island['indv'][island['rank'][0]].adj_matrix)
						rank=np.arange(self.size)
						for i in range(self.size):
								rank[i]=temp[i][0][0][1]
						permute_code=np.arange(0, self.size)
						for i in range(self.size):
								permute_code[i]=temp[i][0][1]
						self.rank_center=island['indv'][island['rank'][0]].adj_matrix
						self.center_generation=generation
				else:
						temp=np.copy(self.rank_center)
						rank=np.arange(self.size)
						for i in range(self.size):
								rank[i]=temp[i][0][0][1]

						permute_code=np.arange(0, self.size)
						for i in range(self.size): 
								permute_code[i]=temp[i][0][1]
                                
                                
                                



				Rvarien=self.kwargs['c1'][0]**(generation-1)*self.kwargs['R_varience_init'][0]
				if Rvarien<self.kwargs['R_varience_LB'][0]:
					Rvarien=self.kwargs['R_varience_LB'][0]
                    
				Pvarien=self.kwargs['c2'][0]**(generation-1)*self.kwargs['P_varience_init'][0]
				if Pvarien<self.kwargs['P_varience_LB'][0]:
					Pvarien=self.kwargs['P_varience_LB'][0]                
                    

				num_rank=self.size
				adjmat_set=[]

				slot_R=np.arange(0,((self.kwargs['slot_R'][0]-1)-1)+1+1,1).astype(float)
				slot_R[slot_R>0]=slot_R[slot_R>0]-0.5

				slot_P=np.arange(0,((self.kwargs['slot_P'][0]-1)-1)+1+1,1).astype(float)
				
				exchange_set=[] 
                
				rank_UB=np.where(rank==self.kwargs['rank']-1)
				rank_LB=np.where(rank==1)



				for samplen in range(self.kwargs['Sample_Numbers'][0]):
									temp=np.random.normal(0,Rvarien,[1,num_rank])
									positive_index=temp>=0
									temp_abs=np.abs(temp)
	
									for i in range(num_rank):
										rtemp=temp_abs[0,i]-slot_R
										rtemp=rtemp>0
										rtemp=rtemp.astype(int)
										rtemp=np.sum(rtemp)
	
										if rtemp>self.kwargs['slot_R'][0]-1:
											rtemp=self.kwargs['slot_R'][0]-1-1 
										else:
											rtemp = rtemp - 1

										if positive_index[0,i]:
											temp[0,i]=rtemp
										else:
											temp[0,i]=-1*rtemp
									temp=temp.astype(int)
                                    
									UB=np.copy(temp[0,rank_UB])   
									LB=np.copy(temp[0,rank_LB])   
          							
									UB=-1*np.abs(UB)
									LB=np.abs(LB)
									temp[0,rank_UB]=UB
									temp[0,rank_LB]=LB 
                                    
                                    
									sampletemp=rank.reshape((1,num_rank)) 
									sampletemp=sampletemp+temp
									sampletemp[sampletemp>(self.rank-1)]=self.rank-1
									sampletemp[sampletemp<1]=1
									samplerank=sampletemp.reshape((num_rank,))

									if np.random.uniform() > Pvarien:
										out=self.out
										rank_temp=samplerank.reshape((self.size,1))
										permute_temp=permute_code.reshape((self.size,1))
										out=out.reshape((self.size,1))
										adj_matrix=np.rec.fromarrays((out, rank_temp))
										adj_matrix=np.rec.fromarrays((adj_matrix, permute_temp))
										adjmat_set.append(adj_matrix)
										exchange_set.append([0])
									else:


										temp=np.random.normal(0,self.kwargs['P_varience_R'][0],[1,1])
										positive_index=temp>=0
										temp_abs=np.abs(temp)



										for i in range(1):
											rtemp=temp_abs[0,i]-slot_P
											rtemp=rtemp>0
											rtemp=rtemp.astype(int)
											rtemp=np.sum(rtemp)

											if rtemp>self.kwargs['slot_P'][0]-1:
												rtemp=self.kwargs['slot_P'][0]-1
											else:
												rtemp = rtemp 
			
											if positive_index[0,i]:
												temp[0,i]=rtemp
											else:
												temp[0,i]=-1*rtemp
										temp=temp.astype(int)
										temp=np.abs(temp)
										samplepermute=np.copy(permute_code)
										ttset=[]
										for tempn in range(int(temp)):
																	tnow=np.arange(0,self.size)
																	np.random.shuffle(tnow)
																	tnow=tnow[0:2]
																	ttset=ttset+tnow.tolist()
																	samplepermute[tnow]=samplepermute[tnow[::-1]]
										
										exchange_set.append(ttset)
										out=self.out
										rank_temp=samplerank.reshape((self.size,1))
										permute_temp=samplepermute.reshape((self.size,1))
										out=out.reshape((self.size,1))
										adj_matrix=np.rec.fromarrays((out, rank_temp))
										adj_matrix=np.rec.fromarrays((adj_matrix, permute_temp))
										adjmat_set.append(adj_matrix)



				for i in range(self.kwargs['Sample_Numbers'][0]): 
										island['indv'][i].adj_matrix=adjmat_set[i]
										island['indv'][i].parents=('%d'%(self.center_generation)+'|'+''.join([str(k) for k in exchange_set[i]]))









		for idx, (k, v) in enumerate(self.societies.items()):
				TN_structure_LocalSampling(v,self.n_generation)



	def __evaluate__(self):

		def score2rank(island, idx):
			sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
			score = island['score']
			sparsity_score = [ s for s, l in score ]
			loss_score = [ l for s, l in score ]

			if 'fitness_func' in self.kwargs.keys():
				if isinstance(self.kwargs['fitness_func'], list):
					fitness_func = self.kwargs['fitness_func'][idx]
				else:
					fitness_func = self.kwargs['fitness_func']
			else:		
				fitness_func = lambda s, l: 1*s+200*l
			
			total_score = [ fitness_func(s, l) for s, l in zip(sparsity_score, loss_score) ]

			island['rank'] = np.argsort(total_score)
			island['total'] = total_score

		# RANKING
		for idx, (k, v) in enumerate(self.societies.items()):
			v['score'] = [ (indv.sparsity ,np.min(indv.repeat_loss)) for indv in v['indv'] ]
			score2rank(v, idx)

	def distribute_indv(self, agent):
		if self.indv_to_distribute:
			indv = self.indv_to_distribute.pop(0)
			if np.log10(indv.sparsity)<1:
				agent.receive(indv)
				self.indv_to_collect.append(indv)
				logging.info('Assigned individual {} to agent {}.'.format(indv.scope, agent.sge_job_id))
			else:
				indv.collect(fake_loss=True)
				logging.info('Individual {} is killed due to its sparsity = {} / {}.'.format(indv.scope, np.log10(indv.sparsity), indv.sparsityB))

	def collect_indv(self):
		for indv in self.indv_to_collect:
			if indv.collect():
				logging.info('Collected individual result {}.'.format(indv.scope))
				self.indv_to_collect.remove(indv)

	def is_finished(self):
		if len(self.indv_to_distribute) == 0 and len(self.indv_to_collect) == 0:
			return True
		else:
			return False

	def get_center(self):
			return self.center_generation,self.rank_center,self.fitness_score_center


class Agent(object):
	def __init__(self, **kwargs):
		super(Agent, self).__init__()
		self.kwargs = kwargs
		self.sge_job_id = self.kwargs['sge_job_id']

	def receive(self, indv):
		indv.deploy(self.sge_job_id)
		with open(base_folder+'/agent_pool/{}.POOL'.format(self.sge_job_id), 'a') as f:
			f.write(evoluation_goal)

	def is_available(self):
		return True if os.stat(base_folder+'/agent_pool/{}.POOL'.format(self.kwargs['sge_job_id'])).st_size == 0 else False

class Overlord(object):
	def __init__(self, max_generation=100, **kwargs):
		super(Overlord, self).__init__()
		self.dummy_func = lambda *args, **kwargs: None
		self.max_generation = max_generation
		self.current_generation = None
		self.previous_generation = None
		self.N_generation = 0
		self.kwargs = kwargs
		self.generation = kwargs['generation']
		self.generation_list = []
		self.available_agents = []
		self.known_agents = {}
		self.time = 0

	def __call_with_interval__(self, func, interval):
		return func if self.time%interval == 0 else self.dummy_func

	def __tik__(self, sec):
		# logging.info(self.time)
		self.time += sec
		time.sleep(sec)

	def __check_available_agent__(self):
		self.available_agents.clear()
		agents = glob.glob(base_folder+'/agent_pool/*.POOL')
		agents_id = [ a.split('/')[-1][:-5] for a in agents ]

		for aid in list(self.known_agents.keys()):
			if aid not in agents_id:
				logging.info('Dead agent id = {} found!'.format(aid))
				self.known_agents.pop(aid, None)

		for aid in agents_id:
			if aid in self.known_agents.keys():
				if self.known_agents[aid].is_available():
					self.available_agents.append(self.known_agents[aid])
			else:
				self.known_agents[aid] = Agent(sge_job_id=aid)
				logging.info('New agent id = {} found!'.format(aid))

	def __assign_job__(self):
		self.__check_available_agent__()
		if len(self.available_agents)>0:
			for agent in self.available_agents:
				self.current_generation.distribute_indv(agent)

	def __collect_result__(self):
		self.current_generation.collect_indv()

	def __report_agents__(self):
		logging.info('Current number of known agents is {}.'.format(len(self.known_agents)))
		logging.info(list(self.known_agents.keys()))

	def __report_generation__(self):
		logging.info('Current length of indv_to_distribute is {}.'.format(len(self.current_generation.indv_to_distribute)))
		logging.info('Current length of indv_to_collect is {}.'.format(len(self.current_generation.indv_to_collect)))
		logging.info([(indv.scope, indv.sge_job_id) for indv in self.current_generation.indv_to_collect])

	def __generation__(self):
		if self.N_generation > self.max_generation:
			return False
		else:
			if self.current_generation is None:
				self.current_generation = self.generation(0,0,0,0,name='generation_init', **self.kwargs)
				self.current_generation.indv_to_distribute = []

			if self.current_generation.is_finished():
				if self.previous_generation is not None:
					self.current_generation(**self.kwargs)
				self.N_generation += 1
				self.previous_generation = self.current_generation
				CG,RC,FC=self.previous_generation.get_center()


				self.current_generation = self.generation(CG,RC,FC,self.N_generation,self.previous_generation,  
														name='generation_{:03d}'.format(self.N_generation), **self.kwargs)

			return True

	def __call__(self):
		while self.__generation__():
			self.__call_with_interval__(self.__check_available_agent__, 4)()
			self.__call_with_interval__(self.__assign_job__, 4)()
			self.__call_with_interval__(self.__collect_result__, 4)()
			self.__call_with_interval__(self.__report_agents__, 180)()
			self.__call_with_interval__(self.__report_generation__, 160)()
			self.__tik__(2)

def score_summary(obj):
	logging.info('===== {} ====='.format(obj.name))

	for k, v in obj.societies.items():
		logging.info('===== ISLAND {} ====='.format(k))

		for idx, indv in enumerate(v['indv']):
			if idx == v['rank'][0]:
				logging.info('\033[31m{} | {:.3f} | {} | {:.5f} | {}\033[0m'.format(indv.scope, np.log10(indv.sparsity), [ float('{:.12f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
				logging.info(indv.adj_matrix)

				dim = indv.adj_matrix.shape[0]
				adj_matrix_T=np.copy(indv.adj_matrix)
				size_data=np.arange(dim)
				for i in range(dim):
								size_data[i]=adj_matrix_T[i][0][0][0]
				rank=np.arange(dim)
				for i in range(dim):
								rank[i]=adj_matrix_T[i][0][0][1]

				permute_code=np.arange(0, dim)
				for i in range(dim):
								permute_code[i]=adj_matrix_T[i][0][1]
				logging.info(permute_code.reshape((1,dim )))
				adj_matrix_R = np.diag(size_data)
				temp=np.arange(dim-1)
				temp=temp[::-1]
				temp[0]=dim-3
				connection_index = []
				connection_index.append(0)
				for i in range(dim):
								if i==1:
										connection_index.append(connection_index[-1]+1)
								else:
										if i==0:
												connection_index.append(connection_index[-1]+(temp[i]+1))
										else:
												connection_index.append(connection_index[-1]+(temp[i-1]+1))
				connection_index=connection_index[0:dim]
				connection =rank.tolist()
				index_tuple=np.triu_indices(dim, 1)
				index_tuple1=index_tuple[0]
				index_tuple2=index_tuple[1]
				index_tuple1=index_tuple1[connection_index]
				index_tuple2=index_tuple2[connection_index]
				index_tuple=[index_tuple1,index_tuple2]
				index_tuple=tuple(index_tuple)
				adj_matrix_R[index_tuple] = connection
				adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]
				index=np.diag_indices_from(adj_matrix_R)
				adj_matrix_R[index]=0

				permute=permute_code

				permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
				for i in range(dim):
										permutation_matrix[permute[i],i] = 1
				adj_matrix_in=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())
				adj_matrix_in[adj_matrix_in==1] = 0
				logging.info(adj_matrix_in)
				logging.info('Parameters:{} '.format(indv.actual_elements))


			else:
				logging.info('{} | {:.3f} | {} | {:.5f} | {}'.format(indv.scope, np.log10(indv.sparsity), [ float('{:.12f}'.format(l)) for l in indv.repeat_loss ], v['total'][idx], indv.parents))
				logging.info(indv.adj_matrix)

				dim = indv.adj_matrix.shape[0]
				adj_matrix_T=np.copy(indv.adj_matrix)
				size_data=np.arange(dim)
				for i in range(dim):
								size_data[i]=adj_matrix_T[i][0][0][0]
				rank=np.arange(dim)
				for i in range(dim):
								rank[i]=adj_matrix_T[i][0][0][1]

				permute_code=np.arange(0, dim)
				for i in range(dim):
								permute_code[i]=adj_matrix_T[i][0][1]
				logging.info(permute_code.reshape((1,dim )))
				adj_matrix_R = np.diag(size_data)
				temp=np.arange(dim-1)
				temp=temp[::-1]
				temp[0]=dim-3
				connection_index = []
				connection_index.append(0)
				for i in range(dim):
								if i==1:
										connection_index.append(connection_index[-1]+1)
								else:
										if i==0:
												connection_index.append(connection_index[-1]+(temp[i]+1))
										else:
												connection_index.append(connection_index[-1]+(temp[i-1]+1))
				connection_index=connection_index[0:dim]
				connection =rank.tolist()
				index_tuple=np.triu_indices(dim, 1)
				index_tuple1=index_tuple[0]
				index_tuple2=index_tuple[1]
				index_tuple1=index_tuple1[connection_index] 
				index_tuple2=index_tuple2[connection_index]
				index_tuple=[index_tuple1,index_tuple2]
				index_tuple=tuple(index_tuple)
				adj_matrix_R[index_tuple] = connection
				adj_matrix_R[np.tril_indices(dim, -1)] = adj_matrix_R.transpose()[np.tril_indices(dim, -1)]  
				index=np.diag_indices_from(adj_matrix_R)
				adj_matrix_R[index]=0

				permute=permute_code

				permutation_matrix=np.zeros(adj_matrix_R.shape,dtype=int)
				for i in range(dim): 
										permutation_matrix[permute[i],i] = 1
				adj_matrix_in=np.diag(size_data)+np.matmul(np.matmul(permutation_matrix,adj_matrix_R),permutation_matrix.transpose())          
				adj_matrix_in[adj_matrix_in==1] = 0
				logging.info(adj_matrix_in)
				logging.info('Parameters:{} '.format(indv.actual_elements))   

if __name__ == '__main__':
	pipeline = Overlord( 
													max_generation=20, generation=Generation, random_init=True, size=6, out=np.array([3,3,3,3,3,3],dtype=int),
         
													# Hyperparameters 
													 rank=8, c1=[0.9],c2=[0.94], fitness_func=[ lambda s, l: s+200*l],Sample_Numbers=[int(sys.argv[2])],
                                                     
                                                    
                                                    R_varience_init=[1],R_varience_LB=[0],slot_R=[6],P_varience_init=[1],P_varience_LB=[0],slot_P=[2],P_varience_R=[0.2],
													# EVALUATION PROPERTIES 
													evaluate_repeat=4, max_iterations=20000, rse_therhold=[1e-12],
                                                    
                                                    
													# ISLAND PROPERTIES 
													N_islands=1,     
													# FOR COMP1.5
													callbacks=[score_summary])
	pipeline()

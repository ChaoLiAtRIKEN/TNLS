import numpy as np, os, sys, re, glob, subprocess, math, unittest, time, shutil, logging, gc, psutil
np.set_printoptions(precision=2)
from tenmul4 import TensorNetwork
from random import shuffle, choice
from itertools import product
from functools import partial
import inspect, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# from overlore import Generation
from pathlib import Path

agent_id = sys.argv[1]
base_folder = './'
try:
	os.mkdir(base_folder+'log')
	os.mkdir(base_folder+'agent_pool')
	os.mkdir(base_folder+'job_pool')
	os.mkdir(base_folder+'result_pool')
except:
	pass

log_name = base_folder + '/log/{}.log'.format(agent_id)
logging.basicConfig(filename=log_name, filemode='a', level=logging.DEBUG,
										format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def evaluate(tf_graph, sess, indv_scope, adj_matrix,therhold,evaluate_repeat, max_iterations, evoluation_goal=None, evoluation_goal_square_norm=None):		
	with tf_graph.as_default():
		with tf.variable_scope(indv_scope):
			dim = adj_matrix.shape[0]
			adj_matrix_T=np.copy(adj_matrix)


			size_data=np.arange(dim)
			for i in range(dim):
				size_data[i]=adj_matrix_T[i][0][0][0]
		
			rank=np.arange(dim)
			for i in range(dim):
				rank[i]=adj_matrix_T[i][0][0][1]

			permute_code=np.arange(0, dim)
			for i in range(dim):
				permute_code[i]=adj_matrix_T[i][0][1]

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


			TN = TensorNetwork(adj_matrix_in)
			output = TN.reduction(False)
			goal = tf.convert_to_tensor(evoluation_goal)

			goal_square_norm = tf.convert_to_tensor(evoluation_goal_square_norm)
			rse_loss = tf.reduce_mean(tf.square(output - goal)) / goal_square_norm
			var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=indv_scope)
			step = tf.train.AdamOptimizer(0.001).minimize(rse_loss, var_list=var_list)
			var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=indv_scope)

		repeat_loss = []
		for r in range(evaluate_repeat):
			sess.run(tf.variables_initializer(var_list))
			for i in range(max_iterations): 
				sess.run(step)
				if sess.run(rse_loss)<therhold:
								break
			repeat_loss.append(sess.run(rse_loss))

	return repeat_loss

def check_and_load(agent_id):
	file_name = base_folder+'/agent_pool/{}.POOL'.format(agent_id)
	if os.stat(file_name).st_size == 0:
		return False, False
	else:
		with open(file_name, 'r') as f:
			goal_name = f.readline()
			evoluation_goal = np.load(goal_name).astype(np.float32)
		return True, evoluation_goal

def memory():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0]/2.**30 
	print('memory use:', memoryUse)

if __name__ == '__main__':
	Path(base_folder+'/agent_pool/{}.POOL'.format(agent_id)).touch()

	while True:
		flag, evoluation_goal = check_and_load(agent_id)
		if flag:
			evoluation_goal_square_norm=np.mean(np.square(evoluation_goal))
			indv = np.load(base_folder+'/job_pool/{}.npz'.format(agent_id))

			scope = indv['scope'].tolist()
			adj_matrix = indv['adj_matrix']
			repeat = indv['repeat']
			iters = indv['iters']
			rse_TH = indv['rse_TH']

			logging.info('Receiving individual {} for {}x{}...'.format(scope, repeat, iters))

			g = tf.Graph()
			sess = tf.Session(graph=g)
			try:
				repeat_loss = evaluate(tf_graph=g, sess=sess, indv_scope=scope, adj_matrix=adj_matrix, therhold=rse_TH,evaluate_repeat=repeat, max_iterations=iters,
																evoluation_goal=evoluation_goal, evoluation_goal_square_norm=evoluation_goal_square_norm)
				logging.info('Reporting result {}.'.format(repeat_loss))
				np.savez(base_folder+'/result_pool/{}.npz'.format(scope.replace('/', '_')),
									repeat_loss=[ float('{:.14f}'.format(l)) for l in repeat_loss ],
									adj_matrix=adj_matrix)

				os.remove(base_folder+'/job_pool/{}.npz'.format(agent_id))
				open(base_folder+'/agent_pool/{}.POOL'.format(agent_id), 'w').close()

			except Exception as e:
				os.remove(base_folder+'/agent_pool/{}.POOL'.format(agent_id))
				raise e

			sess.close()
			tf.reset_default_graph()
			del repeat_loss, g
			gc.collect()

		time.sleep(1)





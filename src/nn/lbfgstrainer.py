#-*- coding: utf-8 -*-
'''
Reordering classifier and related training code

@author: lpeng
'''
from __future__ import division
from sys import stderr
import argparse
import logging
import cPickle as pickle

import codecs

from numpy import concatenate, zeros_like, zeros, save, array, load
from numpy.random import get_state, set_state, seed
from mpi4py import MPI

from ioutil import Writer
from timeutil import Timer
import lbfgs
from ioutil import Reader
from nn.rae import RecursiveAutoencoder
from nn.util import init_W
from nn.instance import Instance
from nn.signals import TerminatorSignal, WorkingSignal, ForceQuitSignal
from errors import GridentCheckingFailedError
from vec.word2vec import Word2VecModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


comm = MPI.COMM_WORLD
worker_num = comm.Get_size()
rank = comm.Get_rank()


def send_terminate_signal():
  param = TerminatorSignal()
  comm.bcast(param, root=0)


def send_working_signal():
  param = WorkingSignal()
  comm.bcast(param, root=0)

  
def send_force_quit_signal():
  param = ForceQuitSignal()
  comm.bcast(param, root=0)


def compute_cost_and_grad(theta, instances, total_internal_node_num,
                           embsize, lambda_reg):
  '''Compute the value and gradients of the objective function at theta
  
  Args:
    theta: model parameter
    instances: training instances
    total_internal_node_num: total number of internal nodes 
    embsize: word embedding vector size
    lambda_reg: the weight of regularizer
    
  Returns:
    total_cost: the value of the objective function at theta
    total_grad: the gradients of the objective function at theta
  '''
  
  if rank == 0:
    # send working signal
    send_working_signal()

    # send theta
    comm.Bcast([theta, MPI.DOUBLE], root=0)
  
    # init recursive autoencoder
    rae = RecursiveAutoencoder.build(theta, embsize)
  
    # compute local reconstruction error and gradients
    rec_error, gradient_vec = process_local_batch(rae, instances)
    
    # compute total reconstruction error
    total_rec_error = comm.reduce(rec_error, op=MPI.SUM, root=0)
    # compute total cost
    reg = rae.get_weights_square()
    total_cost = total_rec_error / total_internal_node_num + lambda_reg/2 * reg

    print('reconstruct error: ' + str(total_cost))

    # compute gradients
    total_grad = zeros_like(gradient_vec)
    comm.Reduce([gradient_vec, MPI.DOUBLE], [total_grad, MPI.DOUBLE],
                op=MPI.SUM, root=0)
    total_grad /= total_internal_node_num
    
    # gradients related to regularizer
    reg_grad = rae.get_zero_gradients()
    reg_grad.gradWi1 += rae.Wi1
    reg_grad.gradWi2 += rae.Wi2
    reg_grad.gradWo1 += rae.Wo1
    reg_grad.gradWo2 += rae.Wo2
    reg_grad *= lambda_reg
    
    total_grad += reg_grad.to_row_vector()

    return total_cost, total_grad
  else:
    while True:
      # receive signal
      signal = comm.bcast(root=0)
      if isinstance(signal, TerminatorSignal):
        return
      if isinstance(signal, ForceQuitSignal):
        exit(-1)
      
      # receive theta
      comm.Bcast([theta, MPI.DOUBLE], root=0)
    
      # init recursive autoencoder
      rae = RecursiveAutoencoder.build(theta, embsize)
    
      # compute local reconstruction error and gradients
      rec_error, gradient_vec = process_local_batch(rae, instances)

      # send local reconstruction error to root
      comm.reduce(rec_error, op=MPI.SUM, root=0)
      
      # send local gradients to root
      comm.Reduce([gradient_vec, MPI.DOUBLE], None, op=MPI.SUM, root=0)
  
 
def process_local_batch(rae, instances):
  gradients = rae.get_zero_gradients()
  total_rec_error = 0
  count = 0
  for instance in instances:
    root_node, rec_error = rae.forward(instance.words_embedding)
    rae.backward(root_node, gradients, freq=instance.freq)
    total_rec_error += rec_error * instance.freq
    count +=1 ;
    if count % 1000 ==0:
      print('finish: ' + str(count))
  print ('\n')
  return total_rec_error, gradients.to_row_vector()


def init_theta(embsize, _seed=None):
  if _seed != None:
    ori_state = get_state()
    seed(_seed)
    
  parameters = []
  
  # Wi1 
  parameters.append(init_W(embsize, embsize))
  # Wi2
  parameters.append(init_W(embsize, embsize))
  # bi
  parameters.append(zeros(embsize))
  
  # Wo1 
  parameters.append(init_W(embsize, embsize))
  # Wo2
  parameters.append(init_W(embsize, embsize))
  # bo1
  parameters.append(zeros(embsize))
  # bo2
  parameters.append(zeros(embsize))

  if _seed != None:  
    set_state(ori_state)
  
  return concatenate(parameters)   


def prepare_data(word2vec_model=None, datafile=None):
  '''Prepare training data
  Args:
    word_vectors: an instance of vec.wordvector
    datafile: location of data file
    
  Return:
    instances: a list of Instance
    total_internal_node: total number of internal nodes
  '''
  instances = []
  total_internal_node = 0
  if rank == 0:
    file = codecs.open(datafile, encoding="UTF-8")
    for line in file.readlines():

      instance = Instance.parse_from_str(line,model=word2vec_model)
      instances.append(instance)
      total_internal_node += instance.number_internal_node
  return instances, total_internal_node




class ThetaSaver(object):
  
  def __init__(self, model_name, every=1):
    self.idx = 1
    self.model_name = model_name
    self.every = every
    
  def __call__(self, xk):
    if self.every == 0:
      return;
    
    if self.idx % self.every == 0:
      model = self.model_name
      pos = model.rfind('.')
      if pos < 0:
        filename = '%s.iter%d' % (model, self.idx)
      else:
        filename = '%s.iter%d%s' % (model[0:pos], self.idx, model[pos:])
      
      with Writer(filename) as writer:
        [writer.write('%20.8f\n' % v) for v in xk]
    self.idx += 1

import os

if __name__ == '__main__':
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-instances', required=True,
  #                     help='instances for training')
  # parser.add_argument('-model', required=True,
  #                     help='model name')
  # parser.add_argument('-word_vector', required=True,
  #                     help='word vector file',)
  # parser.add_argument('-lambda_reg', type=float, default=0.15,
  #                     help='weight of the regularizer')
  # parser.add_argument('--save-theta0', action='store_true',
  #                     help='save theta0 or not, for dubegging purpose')
  # parser.add_argument('--checking-grad', action='store_true',
  #                     help='checking gradients or not, for dubegging purpose')
  # parser.add_argument('-m', '--maxiter', type=int, default=100,
  #                     help='max iteration number',)
  # parser.add_argument('-e', '--every', type=int, default=0,
  #                     help='dump parameters every --every iterations',)
  # parser.add_argument('--seed', default=None,
  #                     help='random number seed for initialize random',)
  # parser.add_argument('-v', '--verbose', type=int, default=0,
  #                     help='verbose level')
  # options = parser.parse_args()

  #
  #
  # instances_file = options.instances
  # model = options.model
  # word_vector_file = options.word_vector
  # lambda_reg = options.lambda_reg
  # save_theta0 = options.save_theta0
  # checking_grad = options.checking_grad
  # maxiter = options.maxiter
  # every = options.every
  # _seed = options.seed
  # verbose = options.verbose


  instances_file = '../mini_corpus.txt'
    #options.instances
  model = '../sample-training-file.mpi-1.model.gz'
    #options.model
  word_vector_file = '../vec/vn_word2vec_model_27/100'
    #options.word_vector
  lambda_reg = 0.15
    #options.lambda_reg
  save_theta0 = 0 #options.save_theta0
  checking_grad = 0 #options.checking_grad
  maxiter = 200
    #= options.maxiter
  every = 0 # options.every
  _seed = None #options.seed
  verbose = 0 #options.verbose
  instances_file_have_embedding = '../mini_corpus_have_embedding.npy'

  
  if rank == 0:
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    if checking_grad:
      logger.setLevel(logging.WARN)
    else:
      logger.setLevel(logging.INFO)
        
    print('Instances file: %s' % instances_file)
    print('Model file: %s' % model)
    print ('Word vector file: %s' % word_vector_file)
    print ('lambda_reg: %20.18f' % lambda_reg)
    print ('Max iterations: %d' % maxiter)
    if _seed:
      print ('Random seed: %s' % _seed)
    print >> stderr, ''
    


       
    print >> stderr, 'preparing data...'
    total_internal_node = 0
    if os.path.exists(instances_file_have_embedding):
      print >> stderr, 'load instances_file_have_embedding ...'
      instances = load(instances_file_have_embedding)
      for instance in instances:
        total_internal_node += instance.number_internal_node

    else:
      print >> stderr, 'load word vectors...'
      word_vectors = Word2VecModel(word_vector_file)
      embsize = word_vectors.embsize
      instances, total_internal_node = prepare_data(word_vectors, instances_file)
      save(instances_file_have_embedding, array(instances))
      print  word_vectors.count_try
      print word_vectors.count_except


    embsize = instances[0].words_embedding.shape[0]
    print >> stderr, 'init. RAE parameters...'
    timer = Timer()
    timer.tic()
    if _seed != None:
      _seed = int(_seed)
    else:
      _seed = None
    print >> stderr, 'seed: %s' % str(_seed)

    theta0 = init_theta(embsize, _seed=_seed)
    theta0_init_time = timer.toc()
    print >> stderr, 'shape of theta0 %s' % theta0.shape
    timer.tic()
    if save_theta0:
      print >> stderr, 'saving theta0...'
      pos = model.rfind('.')
      if pos < 0:
        filename = model + '.theta0'
      else:
        filename = model[0:pos] + '.theta0' + model[pos:]
      with Writer(filename) as theta0_writer:
        pickle.dump(theta0, theta0_writer)
    theta0_saving_time = timer.toc() 
    
    print >> stderr, 'optimizing...'
    
    callback = ThetaSaver(model, every)    
    func = compute_cost_and_grad
    args = (instances, total_internal_node, embsize, lambda_reg)
    theta_opt = None
    try:
      theta_opt = lbfgs.optimize(func, theta0, maxiter, verbose, checking_grad, 
                                 args, callback=callback)
    except GridentCheckingFailedError:
      send_terminate_signal()
      print >> stderr, 'Gradient checking failed, exit'
      exit(-1)

    send_terminate_signal()
    opt_time = timer.toc()

    np_theta_opt = array(theta_opt)
    save('hynguyen_theta_opt',np_theta_opt)

    timer.tic()
    # pickle form
    print >> stderr, 'saving parameters to %s' % model
    with Writer(model) as model_pickler:
      pickle.dump(theta_opt, model_pickler)
    # pure text form
    with Writer(model+'.txt') as writer:
      [writer.write('%20.8f\n' % v) for v in theta_opt]
    thetaopt_saving_time = timer.toc()  
    
    print >> stderr, 'Init. theta0  : %10.2f s' % theta0_init_time
    if save_theta0:
      print >> stderr, 'Saving theta0 : %10.2f s' % theta0_saving_time
    print >> stderr, 'Optimizing    : %10.2f s' % opt_time
    print >> stderr, 'Saving theta  : %10.2f s' % thetaopt_saving_time
    print >> stderr, 'Done!'     
  else:
    # prepare training data
    instances, total_internal_node = prepare_data()
    embsize = instances[0].words_embedding.shape[0]
    param_size = embsize*embsize*4 + embsize*3
    theta = zeros((param_size, 1))    
    compute_cost_and_grad(theta, instances, total_internal_node,
                           embsize, lambda_reg)

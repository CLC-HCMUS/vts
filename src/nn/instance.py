#-*- coding: utf-8 -*-
'''
Training example class
@author: lpeng
'''

class Instance(object):
  '''A reordering training example'''
  
  def __init__(self, words_embedding, number_internal_node, raw_sentence, sentence_embedding = None, freq=1):
    '''
    Args:
      words: numpy.array (an int array of word indices)
      freq: frequency of this training example
    '''
    self.words_embedding = words_embedding
    self.number_internal_node = number_internal_node
    self.raw_sentence = raw_sentence
    self.sentence_embedding = sentence_embedding
    self.freq = freq

  def __str__(self):
    parts = []
    parts.append(' '.join([str(i) for i in self.words]))
    parts.append(str(self.freq))
    return ' ||| '.join(parts)

  @classmethod
  def parse_from_str(cls, line, model):
    words_embedding = model.parseInstanceFromSentence(line)
    return Instance(words_embedding,int(words_embedding.shape[1]) -1,raw_sentence=line,freq=1)

  def to_dictionary(self):
    list_ = {}
    dict['raw_sentence'] = self.raw_sentence
    dict['sentence_embedding'] = self.sentence_embedding.reshape(self.sentence_embedding.size)
    return dict

  def to_list(self):
    return [self.raw_sentence, self.sentence_embedding]

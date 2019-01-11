'''
This is where all the base classes live
'''

class BaseModel(object):
    """
    Base model for tensorflow
    """
    def __init__(self):
        pass

    def build_graph(self):
        '''
        builds the tensorflow graph for the model
        '''
        raise NotImplementedError()

    def add_train_op(self):
        '''
        adds training operations to the graph
        '''
        raise NotImplementedError()

    def encode(self, data):
        '''
        encode input data into a feed dict
        '''
        raise NotImplementedError()

    def train_step(self, sess, fd):
        '''
        proceed forward with one step of training with given feed dict
        '''
        raise NotImplementedError()

    def eval(self, sess, fd):
        '''
        evaluation with given feed dict
        '''
        raise NotImplementedError()

    def save_to(self, sess, path):
        raise NotImplementedError()

    def restore_from(self, sess, path):
        raise NotImplementedError()


class Vocab(object):
    '''
    vocabulary object
    '''

    def __init__(self):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()

    def emb_size(self):
        raise NotImplementedError()

    def word2id(self, w):
        raise NotImplementedError()

    def id2word(self, id):
        raise NotImplementedError()

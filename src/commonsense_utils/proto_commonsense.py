class GraphVertex(object):
    '''
    commonsense graph vertex object
    '''

    def __init__(self, subj, obj, parent, level):
        raise NotImplementedError()

    def add_child(self, c):
        raise NotImplementedError()

    def add_relation(self, r):
        raise NotImplementedError()

    def add_score(self, s):
        raise NotImplementedError()

    def add_path(self):
        raise NotImplementedError()

    def add_total_edges_subj(self, num_edges):
        raise NotImplementedError()

    def add_total_edges_obj(self, num_edges):
        raise NotImplementedError()


class Graph(object):
    '''
    commonsense graph vertex object
    '''

    def __init__(self, data):
        raise NotImplementedError()


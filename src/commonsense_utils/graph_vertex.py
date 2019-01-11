from proto_commonsense import GraphVertex


class Commonsense_GV(GraphVertex):
    '''
    graph vertex object for commonsense graph 
    '''
    
    def __init__(self, subj, obj, parent, level, relation=None):
        self.subj = subj 
        self.obj = obj
        self.relation = []
        if relation: 
            self.relation = [relation] 
        self.parent = parent 
        self.level = level 
        self.children = []
        self.score = 0 
        self.num_paths = 1 
        self.total_edges_subj = 1 
        self.total_edges_obj = 1 
        self.child_scores = []
        self.total_child_list = []
  
    def add_child(self, c):
        self.children.append(c)

    def add_relation(self, r):
        self.relation.append(r)

    def add_score(self, s):
        self.score = s 

    def add_path(self, number_to_add = 0):
        if number_to_add > 0:
            self.num_paths = number_to_add
        else: 
            self.num_paths += 1 

    def add_total_edges_subj(self, num_edges):
        self.total_edges_subj = num_edges 

    def add_total_edges_obj(self, num_edges):
        self.total_edges_obj = num_edges 

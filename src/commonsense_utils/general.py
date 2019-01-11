from __future__ import division
import random 
import re
import time
import copy
import re
import math
import itertools
import numpy as np
from nltk.probability import FreqDist
from graph_vertex import Commonsense_GV
from commonsense_data import COMMONSENSE_REL_LOOKUP
from tqdm import tqdm

def sample_relations(graph, context):
    return sample_relations_top_n(graph, context, 'attention') 


def sample_relations_top_n(graph, context, type_):
    num_total_words = len(context)
    dist = FreqDist(context)
 
    for node in graph:
        node = build_score_per_layer(node, dist, num_total_words)

    for node in graph:
        node = calc_top_n_score_by_level(node)

    for i, node in enumerate(graph):
        graph[i] = prune_graph_by_top_n_softmax(node)

    selected_paths = select_paths(graph) 
    paths = build_subpaths(selected_paths)
    final_paths = list(paths for paths, _ in itertools.groupby(paths))
    random.shuffle(final_paths)
    return final_paths


def get_relations(commonsense_file, hyponyms=False, weights=True): 
    filename = open(commonsense_file)
    relations = dict() 
    for line in filename:
        line = line.split(',')
        w1 = line[0].strip().lower()
        rels = [str(line[1]).strip()]

        for i in range(2, len(line)):
           rels.append(str(line[i].strip().lower()))

        if w1 not in relations:
           relations[w1] = []
           relations[w1].append(rels)
        else:
           relations[w1].append(rels)
    return relations


    
def build_score_per_layer(node, freq_dist, num_total_words): 
    if node.level == 4:
        score = calc_freq(freq_dist, node.obj, num_total_words)
        node.add_score(score)       
        return node

    else:
        if node.level == 3:
            score = calc_PMI(node)
            for c in node.children:
               c = build_score_per_layer(c, freq_dist, num_total_words) 
            node.add_score(score)
            return node

        else: 
            score = calc_freq(freq_dist, node.obj, num_total_words)
            for c in node.children:
               c = build_score_per_layer(c, freq_dist, num_total_words)
            node.add_score(score)
            return node


def calc_top_n_score_by_level(node):
    if not node.children:
        return node 
    else:
        for i, c in enumerate(node.children): # updates all children to hold correct scores
            node.children[i] = calc_top_n_score_by_level(c)
             
        
        scores = []

        for i, c in enumerate(node.children): 
            scores.append(c.score)
        softmaxed_scores = softmax(scores)
    
        for i, c in enumerate(node.children): # all the children have been updated
            if c.children:
               np_child_scores = np.array(c.child_scores)
               if c.level == 1:
                   top_n = 2
               elif c.level == 2:
                   top_n = 2
               elif c.level == 3:
                   top_n = 1
               elif c.level == 4:
                   top_n = 2
 
               highest_idx = np_child_scores.argsort()[-top_n:][::-1] 
               new_scores = np.array([c.child_scores[j] for j in highest_idx])
	       softmaxed_scores[i] += np.mean(new_scores) 
        
        node.child_scores = softmaxed_scores

        return node 



def prune_graph_by_top_n_softmax(node): 
    if not node.children:
        return node
    else:
        np_child_scores = np.array(node.child_scores)
        if node.level == 1:
            top_n = 2
        elif node.level == 2:
            top_n = 2
        elif node.level == 3:
            top_n = 1
        elif node.level == 4:
            top_n = 2

        highest_idx = np_child_scores.argsort()[-top_n:][::-1] 
	
        for i,c in enumerate(node.children):
            if i not in highest_idx:
                node.children[i] = None
        for i, n in enumerate(node.children):
            if n!= None:
                node.children[i] = prune_graph_by_top_n_softmax(node.children[i])
        return node


def calc_freq(freq_dist, obj, num_total_words):
    if '_' not in obj: 
        score = float(freq_dist[obj]/num_total_words)
    else:
        s = []
        for o in obj.split('_'):
            s.append( float(freq_dist[o]/num_total_words) )
        score = min(s) 
    return score


def calc_PMI(node):
    num_paths = node.num_paths 
    p_subj_obj = num_paths
    p_obj = node.total_edges_obj
    total_paths = 0
    count = 0
    while node.parent != None:
        total_paths += (node.num_paths - 1) #counts only extra paths
        node = node.parent
    if total_paths == 0: 
        p_subj = 1
    else:
        p_subj = total_paths
        
    pmi = math.log((p_subj_obj + 1)/(p_obj*p_subj +1))
    n_pmi = pmi/(-math.log(p_subj_obj + 1))
    return n_pmi 


def sublist(lst1, lst2):
    lst1 = ' '.join(lst1)
    if lst1 in lst2:
        return True
    return False


def avg(lst):
  avg = sum(lst) / len(lst)
  return avg


def softmax(x):
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()


def select_paths(graph):
    selected_paths = []
    for g in graph:
        if not g.children:
            path = get_path(g)
            selected_paths.append(path)
        elif g.children:
            for c in g.children:
                if c!= None and not c.children:
                    path = get_path(c) 
                    selected_paths.append(path)
                elif c!= None and c.children:
                    for c_i in c.children:
                        if c_i!= None and not c_i.children:
                            path = get_path(c_i) 
                            selected_paths.append(path)
                        elif c_i!= None and c_i.children:
                            for c_ij in c_i.children:
                                if c_ij != None:
                                    path = get_path(c_ij) 
                                    selected_paths.append(path)
    return selected_paths


def get_path(node):
    path = []
    level = 0
    while node!= None:
        tup = (node.subj, node.relation[0], node.obj)
        path.append(tup)
        node = node.parent
        level += 1
   
    path = path[::-1]
    return path

 
def build_subpaths(paths):
    selected_paths = []

    for s in paths:
        sel_rel = []
        for i in range(1, len(s)+1):           
            sel_rel.append(s[:i])
        selected_paths.extend(sel_rel)

    final_paths = []
    for p in selected_paths:
        path = to_rel_lst(p)
        if path:
            final_paths.append(path)

    final_paths.sort()
    return final_paths


def to_rel_lst(relations):
    r_0 = relations[0][0].replace('_', ' ').split()
      
    final_relations = r_0
    for r in relations:
        r_1 = [COMMONSENSE_REL_LOOKUP[r[1]]] 
        r_2 = r[2].replace('_', ' ').split()
        final_relations.extend(r_1)
        final_relations.extend(r_2)

    return final_relations
 

def is_new_vertex(graph, curr_subj, curr_obj, curr_rel, level, parent_vertex):
    new_vertex = True
    for g in graph:
        if level == 1:
            if g.subj == curr_subj and g.obj == curr_obj:  
                g.add_path()
                g.add_relation(curr_rel) 
                new_vertex = False 
        if level == 2:
            for c in g.children:
                if c.subj == curr_subj and c.obj == curr_obj:
                    c.add_path()
                    c.add_relation(curr_rel)
                    new_vertex = False

        if level == 3:
            for c in g.children:
                for c_i in c.children:
                    if c_i.subj == curr_subj and c_i.obj == curr_obj:
                        c_i.add_path()
                        c_i.add_relation(curr_rel)
                        new_vertex = False
        if level == 4:
             for c in g.children:
                 for c_i in c.children:
                    for c_ij in c_i.children:
                        if c_ij.subj == curr_subj and c_ij.obj == curr_obj:
                            c_ij.add_path()
                            c_ij.add_relation(curr_rel)
                            new_vertex = False

    if parent_vertex != None:
        for c in parent_vertex.children:
            if c.subj == curr_subj and c.obj == curr_obj:
                c.add_path()
                new_vertex = False
                if curr_rel not in c.relation:
                    c.add_relation(curr_rel)

    return new_vertex, graph, parent_vertex

 
def check_context(obj, subj, context_str, context, freq_words):
    in_context = False
    if '_' in obj and subj != obj:
        in_context = sublist(obj.split('_'), context_str)
    elif (obj in context) and (obj not in freq_words) and (subj != obj):
        in_context = True
    return in_context


def create_vertex(subj, obj, parent, level, rel, definitions):
    vertex = Commonsense_GV(subj, obj, parent, level, rel) 
    vertex.add_total_edges_subj(len(definitions[subj]))
    if obj in definitions:
        vertex.add_total_edges_obj(len(definitions[obj]))
    return vertex


def check_new_vertex(graph, subj, obj, rel, level, parent, definitions):
    new_vertex, graph, parent = is_new_vertex(graph, subj, obj, rel, level, parent)
    if new_vertex:
        vertex = create_vertex(subj, obj, parent, level, rel, definitions) 
        return vertex, graph, parent
    return None, graph, parent

def prune_freq(dist, word):
    if not '_' in word and dist[word] < 2:
        return True
    return False
        
def prune_PMI(total_edges, vertex):
    if total_edges > 500: #aggressive pruning
       score = calc_PMI(vertex)
       if score < 8:
           return True
    return False


def build_trees(definitions, query, freq_words, context):
    context_string = ' '.join(context)
    num_total_words = len(context)
    query = [q.lower() for q in query if (q not in freq_words and q in definitions)]
    dist = FreqDist(context)
    graph = []

    total_num_edges = 0

    for q in query:
        for (rel, w_2) in definitions[q]:
            if check_context(w_2, q, context_string, context, freq_words):
                new_vertex_1, graph, parent_vertex = is_new_vertex(graph, q, w_2, rel, 1, None)
                total_num_edges += 1 
                if new_vertex_1:
                    vertex_1 = create_vertex(q, w_2, None, 1, rel, definitions)

                if w_2 in definitions:
                    for (rel_2, w_3) in definitions[w_2]:
                        if check_context(w_3, w_2, context_string, context, freq_words) \
                            and not prune_freq(dist, w_3):
                            vertex_2, graph, vertex_1 = check_new_vertex(graph, w_2, w_3, rel_2, 2, \
                                vertex_1, definitions)
                            total_num_edges += 1 
                            if not vertex_2:
                                continue     

                            if w_3 in definitions:
                                for (rel_3, w_4) in definitions[w_3]:
                                    if (w_4 not in freq_words) and (w_3 != w_4):
                                        vertex_3, graph, vertex_2 = check_new_vertex(graph, w_3,\
                                            w_4, rel_3, 3, vertex_2, definitions)
                                        total_num_edges += 1 
                                        if not vertex_3 or prune_PMI(total_num_edges, vertex_3):
                                            continue

                                        if w_4 in definitions:  
                                            for (rel_4, w_5) in definitions[w_4]:
                                                if check_context(w_5, w_4, context_string, context, \
                                                    freq_words):
                                                    vertex_4, graph, vertex_3 = check_new_vertex(graph,\
                                                         w_4, w_5, rel_4, 4, vertex_3, definitions)
                                                    total_num_edges += 1 
                                                    if not vertex_4:
                                                         continue
                                                    vertex_3.add_child(vertex_4)	
                                        vertex_2.add_child(vertex_3)
                            vertex_1.add_child(vertex_2)
                if new_vertex_1:
                    graph.append(vertex_1)
    return graph

def build_trees_one_hop(definitions, query, freq_words, context):
    context_string = ' '.join(context)
    num_total_words = len(context)
    query = [q.lower() for q in query if (q not in freq_words and q in definitions)]
    dist = FreqDist(context)
    graph = []

    for q in query:
        for (rel, w_2) in definitions[q]:
            if check_context(w_2, q, context_string, context, freq_words):
                new_vertex_1, graph, parent_vertex = is_new_vertex(graph, q, w_2, rel, 1, None)
                if new_vertex_1:
                    vertex_1 = create_vertex(q, w_2, None, 1, rel, definitions)
                    graph.append(vertex_1)
    return graph



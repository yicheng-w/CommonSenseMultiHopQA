from sys import argv
import numpy as np

from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu

meteor_obj = Meteor()
rouge_obj = Rouge()
cider_obj = Cider()
bleu_obj = Bleu(4)



ref1 = argv[1]
ref2 = argv[2]
system = argv[3]

ref1_strs = []
ref2_strs = []
sys_strs = []

with open(ref1, 'r') as f:
    for line in f:
        ref1_strs.append(line.strip())


with open(ref2, 'r') as f:
    for line in f:
        ref2_strs.append(line.strip())


with open(system, 'r') as f:
    for line in f:
        sys_strs.append(line.strip())


assert len(ref1_strs) == len(ref2_strs)
assert len(ref2_strs) == len(sys_strs)

word_target_dict = {}
word_response_dict = {}

rouges = []
for i in range(len(ref1_strs)):
    wtd = {i: [ref1_strs[i], ref2_strs[i]]}
    wrd = {i: [sys_strs[i]]}
    rouge, _ = rouge_obj.compute_score(wtd, wrd)

    rouges.append(rouge)

print(np.mean(rouges))

with open("%s-rouges.txt" % system, 'w') as outf:
    for r in rouges:
        outf.write(str(r)+'\n')

for i in range(len(ref1_strs)):
    word_target_dict[i] = [ref1_strs[i], ref2_strs[i]]
    word_response_dict[i] = [sys_strs[i]]

bleu_score, bleu_scores = bleu_obj.compute_score(
        word_target_dict, word_response_dict)
bleu1_score, _, _, bleu4_score = bleu_score
bleu1_scores, _, _, bleu4_scores = bleu_scores
meteor_score, meteor_scores = meteor_obj.compute_score(
        word_target_dict, word_response_dict) 
rouge_score, rouge_scores = rouge_obj.compute_score(
        word_target_dict, word_response_dict) 
cider_score, cider_scores = cider_obj.compute_score(
        word_target_dict, word_response_dict) 

print("ROUGE-L: ", rouge_score)
print("BLEU-1: ", bleu1_score)
print("BLEU-4: ", bleu4_score)
print("METEOR: ", meteor_score)
print("CiDER: ", cider_score)

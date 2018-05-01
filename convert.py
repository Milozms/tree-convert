import nltk
from nltk.tree import Tree
import linecache
from tqdm import tqdm

head_rules_0 = {
	'ADJP'	: [
		('r', ['ADJP', 'JJ', 'AD'])
	],
	'ADVP'	: [
		('r', ['ADVP', 'AD', 'CS', 'JJ', 'NP', 'PP', 'P', 'VA', 'VV'])
	],
	'CLP'	: [
		('r', ['CLP', 'M', 'NN', 'NP'])
	],
	'CP'	: [
		('r', ['CP', 'IP', 'VP'])
	],
	'DNP'	: [
		('r', ['DEG', 'DNP', 'DEC', 'QP'])
	],
	'DP'	: [
		('r', ['M']),
		('l', ['DP', 'DT', 'OD'])
	],
	'DVP'	: [
		('r', ['DEV', 'AD', 'VP'])
	],
	'IP'	: [
		('r', ['VP', 'IP', 'NP',])
	],
	'INTJ'	: [
		('r', ['INTJ', 'IJ']) # add
	],
	'LCP'	: [
		('r', ['LCP', 'LC'])
	],
	'LST'	: [
		('r', ['CD', 'NP', 'QP'])
	],
	'NP'	: [
		('r', ['NP', 'NN', 'IP', 'NR', 'NT'])
	],
	'NN'	: [
		('r', ['NP', 'NN', 'IP', 'NR', 'NT'])
	],
	'PP'	: [
		('l', ['P', 'PP'])
	],
	'PRN'	: [
		('l', ['PU'])
	],
	'QP'	: [
		('r', ['QP', 'CLP', 'CD'])
	],
	'TYPO'	: [
		('r', ['VP', 'IP', 'NP']) # add
	],
	'UCP'	: [
		('l', ['IP', 'NP', 'VP'])
	],
	'VCD'	: [
		('l', ['VV', 'VA', 'VE'])
	],
	'VCP': [
		('r', ['VCP', 'VV', 'VA', 'VC', 'VE'])  # add
	],
	'VNV'	: [
		('r', ['VNV', 'VV', 'VA', 'VC', 'VE'])
	],
	'VP'	: [
		('l', ['VE', 'VC', 'VV', 'VNV', 'VPT', 'VRD', 'VSB', 'VCD', 'VP', 'VA', 'BA', 'LB']) # add VA BA LB
	],
	'VPT'	: [
		('l', ['VA', 'VV'])
	],
	'VRD'	: [
		('l', ['VV', 'VA'])
	],
	'VSB'	: [
		('r', ['VV', 'VE'])
	],
	'FRAG'	: [
		('r', ['VV', 'NR', 'NN', 'NT'])
	]
}

head_rules = {
	'ADJP'	: [
		('r', ['ADJP', 'JJ']),
		('r', ['AD', 'NN', 'CS'])
	],
	'ADVP'	: [
		('r', ['ADVP', 'AD', 'CS', 'JJ', 'NP', 'PP', 'P', 'VA', 'VV'])
	],
	'CLP'	: [
		('r', ['CLP', 'M', 'NN', 'NP'])
	],
	'CP'	: [
		('r', ['DEC', 'SP']),
		('l', ['ADVP', 'CS']),
		('r', ['CP', 'IP', 'VP'])
	],
	'DNP'	: [
		('r', ['DEG', 'DNP']),
		('r', ['DEC', 'QP'])
	],
	'DP'	: [
		('r', ['M']),
		('l', ['DP', 'DT', 'OD'])
	],
	'DVP'	: [
		('r', ['DVP', 'DEV', 'AD', 'VP'])
	],
	'IP'	: [
		('r', ['VP', 'IP', 'NP']),
		('r', ['VV'])
	],
	'INTJ'	: [
		('r', ['INTJ', 'IJ']) # add
	],
	'LCP'	: [
		('r', ['LCP', 'LC'])
	],
	'LST'	: [
		('l', ['LST', 'CD', 'OD', 'NP', 'QP'])
	],
	'NP'	: [
		('r', ['NP', 'NN', 'IP', 'NR', 'NT', 'QP'])
	],
	'NN'	: [
		('r', ['NP', 'NN', 'IP', 'NR', 'NT'])
	],
	'PP'	: [
		('l', ['P', 'PP'])
	],
	'PRN'	: [
		('r', ['NP', 'IP', 'VP', 'NT', 'NR', 'NN', 'PU'])
	],
	'QP'	: [
		('r', ['QP', 'CLP', 'CD', 'OD'])
	],
	'TYPO'	: [
		('r', ['VP', 'IP', 'NP']) # add
	],
	'UCP'	: [
		('r', ['IP', 'NP', 'VP'])
	],
	'VCD'	: [
		('r', ['VCD', 'VV', 'VA', 'VC', 'VE'])
	],
	'VCP': [
		('r', ['VCP', 'VV', 'VA', 'VC', 'VE'])  # add
	],
	'VNV'	: [
		('r', ['VNV', 'VV', 'VA', 'VC', 'VE'])
	],
	'VP'	: [
		('l', ['VE', 'VC', 'VV', 'VNV', 'VPT', 'VRD', 'VSB', 'VCD', 'VP', 'VA', 'BA', 'LB', 'VCP']) # add VA BA LB
	],
	'VPT'	: [
		('r', ['VNV', 'VA', 'VV', 'VC', 'VE'])
	],
	'VRD'	: [
		('r', ['VRD', 'VV', 'VA', 'VC', 'VE'])
	],
	'VSB'	: [
		('r', ['VSB', 'VV', 'VA', 'VC', 'VE'])
	],
	'FRAG'	: [
		('r', ['VV', 'NR', 'NN', 'NT'])
	]
}

def match_rule(tree):
	'''
	Return the first constituent that match the rule.
	:param tree: a phrase tree
	:return: the index of head subnode
	'''
	if len(tree) == 1:
		return 0
	label = tree.label().split('-')[0]
	if label not in head_rules:
		print('Error: no rule for label %s' % label)
		return None
	rules = head_rules[label]
	for rule in rules:
		direction = rule[0]
		sequence = rule[1]
		if direction == 'l':
			for idx in range(len(tree)):
				if tree[idx].label() in sequence:
					return idx
		else:
			for idx in range(len(tree) - 1, -1, -1):
				if tree[idx].label() in sequence:
					return idx
	if rules[-1][0] == 'l':
		return 0
	else:
		return len(tree) - 1

def find_head(tree):
	'''
	Recursively find head word of each sub tree, and set the label as head word
	:param tree:
	:return: head word
	'''
	if tree.height() < 2:
		print('Error: height error.')
		return None
	if tree.height() == 2:
		# terminal
		tree.set_label(tree[0])
		return tree[0]
	head_idx = match_rule(tree)
	for node in tree:
		node.set_label(find_head(node))
	head_word = tree[head_idx].label()
	tree.set_label(head_word)

	return head_word


def find_dependency_parent(tree, leave_idx):
	'''
	find dependency parent for each leave
	:param leave position:
	:return: parent index (in sentence)
	'''
	for depth in range(len(leave_idx) - 1, -1, -1):
		parent_idx = leave_idx[:depth]
		if tree[parent_idx].label() != tree[leave_idx]:
			return tree[parent_idx].label()
	return (0, 'root')

def convert(tree):
	'''
	Convert phrase tree to dependency tree
	:param tree:
	:return: list of dependency parent index
	'''
	for idx, l in enumerate(tree.treepositions('leaves')):
		tree[l] = (idx + 1, tree[l])
	# tree.draw()
	find_head(tree)
	dependency_parents = []
	for leave_idx in tree.treepositions('leaves'):
		# for each leave, find its dependency head
		dependency_parents.append(find_dependency_parent(tree, leave_idx))
	return dependency_parents


def main(infile = './ctb.bracketed', outfile = './ctb-convert.conll'):
	with open(outfile, 'w') as outf:
		for idx, line in enumerate(tqdm(linecache.getlines(infile))):
			if idx % 2 == 0:
				outf.write(line)
			else:
				tree_str = line.strip('\n')
				tree = Tree.fromstring(tree_str)
				pos_list = tree.pos()
				dependency_parents = convert(tree)
				for idx, pos_word in enumerate(pos_list):
					word, pos = pos_word
					outf.write('%d\t%s\t_\t_\t%s\t_\t%s\tX\t_\t_\n' % (idx+1, word, pos, dependency_parents[idx][0]))
				outf.write('\n')



def test():
	tree = Tree.fromstring('(IP (IP (NP-SBJ (QP (CD 许多)) (ADJP (JJ 司处长级)) (NP (NN 官员))) (VP (PP-TMP (P 在) (LCP (IP (VP (VV 谈及) (NP-OBJ (DNP (NP (PN 自己)) (DEG 的)) (NP (NN 出身))))) (LC 时))) (PU ，) (LCP-LOC (NP (DNP (NP-PN (NR 台湾)) (DEG 的)) (NP (NN 学历))) (LC 之外)) (PU ，) (ADVP (AD 也)) (ADVP (AD 不)) (VP (VV 忘) (IP-OBJ (VP (VV 加) (NP-OBJ (QP (CLP (M 个))) (NP (NP (DNP (PU 「) (NP (NN 本土)) (DEG 的) (PU 」)) (NP-PN (NR 澳门) (NN 大学))) (CC 或是) (NP (NP (NP (DNP (NP-PN (NR 葡国)) (DEG 的)) (NP-PN (NR 里斯本))) (CC 或) (NP (NN 大陆))) (NP (NN 学历)))))))))) (PU ，) (IP (IP-TPC-3 (PU 「) (NP-SBJ (PN 这)) (VP (VV 叫) (NP-OBJ (NN 漂白)))) (PU ，) (PU 」) (NP-SBJ (CP (CP (IP (VP (VP (PP-LOC (P 在) (NP-PN (NR 澳门))) (VP (VCP (VV 出生) (VV 长大)))) (PU ，) (VP (PP-MNR (P 以) (NP-TTL (PU 〈) (NP (NP-PN (NR 澳门)) (NP (NN 主权) (NN 问题))) (NP (NN 始末)) (PU 〉))) (VP (VV 为) (NP-OBJ (NN 博士) (NN 论文)))))) (DEC 的))) (NP-APP (NP-PN (NR 中国) (NN 时报)) (CP (IP (VP (VV 驻) (NP-PN-OBJ (NR 港) (NR 澳))))) (NP (NN 主笔))) (NP-PN (NR 谭志强))) (VP (DVP (VP (VV 开) (NP-OBJ (NN 玩笑))) (DEV 地)) (VP (VV 说)))) (PU 。))')
	dlist = convert(tree)
	print(dlist)
	tree.draw()

if __name__ == '__main__':
	# test()
	main()
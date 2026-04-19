#评估指标
import difflib

def sequence_similarity(a,b):
	""" 基本文本相似度"""
	return difflib.SequenceMatcher(None,a,b).ratio()

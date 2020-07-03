from nltk.translate.bleu_score import corpus_bleu
def bleu(candidates, references):
	score = corpus_bleu(references, candidates)
	return score
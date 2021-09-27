- How to run from command line: "python3 score.py pred_results.txt" where pred_results.txt contains the NER of each word on each line in the format "<spanish word whose NER label is to be predicted> <gold-standard label> <predicted label>". score.py is the evaluation script provided.
- Sample output:
	processed 51533 tokens with 3558 phrases; found: 3461 phrases; correct: 2712.
	accuracy:  97.10%; precision:  78.36%; recall:  76.22%; FB1:  77.28
	              LOC: precision:  80.26%; recall:  74.26%; FB1:  77.14  1003
	             MISC: precision:  72.91%; recall:  53.98%; FB1:  62.03  251
	              ORG: precision:  75.40%; recall:  80.36%; FB1:  77.80  1492
	              PER: precision:  83.78%; recall:  81.50%; FB1:  82.62  715
- Evaluation scripts has been reused from http://computational-linguistics-class.org/homework/ner/ner.html
-We use F1-score (primarily) but also evaluate recall and precision and secondary evaluation metrics.
- F1 score metric: 2*Precision*Recall/(Precision+Recall) will be used where 
Precision = (# of correct NER label output)/(# of correct NER label output + sum across all labels(# of times a particular label was predicted but it was incorrect)) = tp/(tp+fp).
Recall = (# of correct NER label output)/(# of correct NER label output + sum across all labels(# of times a particular label was NOT predicted but it was the correct gold standard)) = tp/(tp+fn).
-Recall is the ratio of the number of correctly labeled responses to the total that should have been labeled; Precision is the ratio of the number of correctly labeled responses to the total labeled; and F-measure is the harmonic mean of the two. 
- This metric score is used in cross-lingual NER papers- https://arxiv.org/pdf/1808.09861.pdf and is usually used for NER task as shown here- https://en.wikipedia.org/wiki/Named-entity_recognition and section 18.1.5 of Dr. Jurafsky's book-https://web.stanford.edu/~jurafsky/slp3/18.pdf
- Higher the F1-score the better the model is at performing NER task.

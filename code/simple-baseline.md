# Simple Baseline

For our simple baseline, we mainly rely on the information that can be transferred across languages using cross-lingual word embeddings. For cross-lingual embeddings we use fastText word embeddings for English and Spanish, mapped onto the same space using the supervised, iterative Procrustes method introduced in (Conneau et al., 2017) \cite{conneau2017word}. After obtaining similar vectors, our simple baseline works by training a Logistic Regression classifier on the English training set using only the embeddings of the previous, current, and next words, if they exist. Then we directly run this classifier on the Spanish test set. Our classifier is thus truly simple and only relies on the mapped word embeddings for any transfer of knowledge across languages.

Here is some sample output from our classifier on the Spanish test set:

```
Por	O	O
Mario	B-PER	B-PER
Etchart	I-PER	I-PER
Sydney	B-LOC	I-PER
(	O	O
Australia	B-LOC	B-LOC
)	O	O
,	O	O
23	O	O
may	O	O
(	O	O
EFE	B-ORG	B-ORG
)	O	O
.	O	O
```

The code for our simple baseline is provided in simple-baseline.py. Note that this code was run on google colab, so it may have some features such as the ! command line and the mounting of the Google Drive that are specific to colab (these are just the first 5 lines and should be removed). Also, the directories need to be changed. The magnitude files we use were converted from the MUSE mapped English and Spanish fastTest embeddings, and we have filtered them to include only the English words in the training set and the Spanish words in the dev and test set, respectively. These files are available here: https://drive.google.com/open?id=1EY0SD7N0wO2wuYgC0pxQeGdIGg_N7bGG, https://drive.google.com/open?id=1kTfazcRjdZZqRjkIBgoPri91GlMfAKL4. Note that even after filtering they are fairly large so we could not include them with our submission.

Evaluating our results with our scoring script gives the following results:

```
processed 316248 tokens with 22355 phrases; found: 24189 phrases; correct: 9013.
accuracy:  91.18%; precision:  37.26%; recall:  40.32%; FB1:  38.73
              LOC: precision:  47.68%; recall:  51.21%; FB1:  49.38  6441
             MISC: precision:   4.42%; recall:   5.33%; FB1:   4.84  3030
              ORG: precision:  33.40%; recall:  36.00%; FB1:  34.65  9473
              PER: precision:  50.41%; recall:  52.29%; FB1:  51.33  5245
```

Thus we get an average F1 score of 37.63 on named entities, performing the worst on MISC entities, which makes sense as there is the most variability among entities for MISC types, and there should thus be more variability among word embeddings for these types as well.

<!--
Bibtext link to conneau article

@article{conneau2017word,
  title={Word translation without parallel data},
  author={Conneau, Alexis and Lample, Guillaume and Ranzato, Marc'Aurelio and Denoyer, Ludovic and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:1710.04087},
  year={2017}
}

-->

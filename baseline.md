# Published Baseline:

## Step 1: Producing translations using the CSLS Metric

First, we need to run CSLSTranslation.ipynb. 

DES: We follow the first step described in our main paper \cite{xie2018neural} and produce translations by using the CSLS metric to find the closest target word in the aligned space.

INPUT: We need to have aligned magnitude files on a google drive that we can access, that contains vectors for all the words in the English training set and Spanish test set. Links to sample vectors (produced using the MUSE library, and downloaded fastText vectors), can be found here: https://drive.google.com/open?id=1Ht1MEqd-rsbzqQ3s7DBA2MVm2JISLMh-, https://drive.google.com/open?id=11tWyhf2Ww0jExCpvmkzEhz0dodmHUbeD.

(For example, after downloading MUSE, run:
```
python supervised.py --src_lang en --tgt_lang es --src_emb wiki.en.vec --tgt_emb wiki.es.vec --n_refinement 3 --dico_train identical_char --max_vocab 100000
```
The authors of the paper also use this library, and do not produce new code for this section. As we are reimplementing their model, we do not need to rewrite code for mapping bilingual vectors. We also use magnitude to filter the English words for the words only contained in the data, to speed up runtime. As noted in the paper, we cut the Spanish words off at a vocabulary of 100,000. Not needed, but sample code can be found here: https://drive.google.com/open?id=1h52JSS_3tdxuYpYQAH-oAz5L7OMKywft)

OUTPUT: translations_bi.txt (includes English word and Spanish translation on each line)

##Step 2 setting up the architecture, loading data, training and evaluating:
Open published_baseline_reimplemented in google colab. The notebook has been divided into different sections. We are leaving our output intact so it can be seen by the grading TA/professor. If you want to be able to run it yourself you would need access to our shared drive where all the data is. Kindly email Aashish Singh- saashish@seas.upenn.edu and he will grant you access to the drive post which you can run it yourself. 
If you are running it yourself make sure to run the cells sequentially. 
DES: We follow the architecture setup as defined in our main paper in our main paper \cite{xie2018neural} and produce a spanish NERv deep learning model with F1 score of ~0.59 (+- 0.1) on validation data.

Input: translations_bi.txt, conll2002 
train_paths = ['Data/eng.train', 'Data/eng.testa', 'Data/eng.testb']
val_paths = ['Data/esp.testa']
test_paths = ['Data/esp.train', 'Data/esp.testb'],
spanish.glove.gigaword_wiki.100d.magnitude

Output: model, results.txt showing on each line: 
[spanish word    gold standard label    predicted label]
(model has Best F1 score on validation data.)






<!--
Bibtext link to main article

@article{xie2018neural,
  title={Neural cross-lingual named entity recognition with minimal resources},
  author={Xie, Jiateng and Yang, Zhilin and Neubig, Graham and Smith, Noah A and Carbonell, Jaime},
  journal={arXiv preprint arXiv:1808.09861},
  year={2018}
}


-->

# Adaptil
Investigating the Domain Robustness of Distilled Models and Pre-training models. 
Most of the experiment in this repo was presented in [Pretrained Transformers Improve Out-of-Distribution Robustness](https://arxiv.org/abs/2004.06100). Did not train models for few tasks due to minimal shift in distribution.

## Tasks and Domains:
- Sentiment Analysis (IMDB, SST2)
- Natural Language Inference (MNLI Matching Domains/MNLI-HANS)
- Paraphase Indentification(QQP, PAWS)


## Representations:

### Amazon Product Reviews
![alt text](/assets/sa-amazon-review-rep-analysis.png "Amazon Representation")

### Amazon Product Reviews
![alt text](/assets/mnli-rep-analysis.png "Paraphrase IID vs OOD")

### Paraphrase Indentification (QQP-PAWS)
![alt text](/assets/paraphrase-qqp-paws.png "Paraphrase Representation")

### WILDS Toxic Comments
![alt text](/assets/toxic-comments-rep-analysis.png "WILDS Representation")


### Sentiment Analysis (IMDB-SST2)
![alt text](/assets/sa-imdb_sst2.png "Sentiment Analysis Representation")

## Results (IID-vs-IID and Performance Drop(PD)):

### Paraphrase (QQP-PAWS)

 ![alt text](/assets/paraphrase/iid-vs-ood.png "Paraphrase IID vs OOD")
 ![alt text](/assets/paraphrase/pd.png "Paraphrase IID vs OOD")

### MNLI (Matched Domain)
![alt text](/assets/mnli/iid-vs-ood.png "Paraphrase IID vs OOD")
![alt text](/assets/mnli/pd.png "MNLI IID vs OOD")


### Sentiment Analysis (IMDB-SST2)

 ![alt text](/assets/imdb_sst2-sa/iid-vs-ood.png "Paraphrase IID vs OOD")
 ![alt text](/assets/imdb_sst2-sa/pd.png "Sentiment Analysis IID vs OOD")









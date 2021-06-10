# Mitigating Unwanted Biases in Word Embeddings with Adversarial Learning using PyTorch
PyTorch implementation of "Mitigating Unwanted Biases in Word Embeddings with Adversarial Learning". Adapted from https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb, which is written with TensorFlow.

Large parts of the data processing code and documentation are copied directly from https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb. Both https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb and this repository implement an experiment from "[Mitigating Unwanted Biases with Adversarial Learning](https://arxiv.org/pdf/1801.07593.pdf)". One way in which this code differs from the original implementation is that it uses two-means to compute the binary gender bias direction instead of PCA.

"questions-words.txt" may be found at https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt, and "GoogleNews-vectors-negative300.bin" may be found at https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz. This code depends on `torch`, `gensim`, and `allennlp`, and requires Python3.

To run this code, simply execute `python3 adversarial_bias_mitigation.py`.




# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

import torch
from gensim.models.keyedvectors import EuclideanKeyedVectors
from allennlp.fairness.bias_direction import TwoMeansBiasDirection
from utils import *

WORD2VEC_FILE = "GoogleNews-vectors-negative300.bin"
ANALOGIES_FILE = "questions-words.txt"
WORD = "she"

# load Google News word embeddings using gensim
client = EuclideanKeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=True)
analogies = load_analogies(ANALOGIES_FILE)

"""
Mitigating unwanted biases with adversarial learning based on Zhang, B.H., Lemoine, B., & Mitchell, M. (2018). [Mitigating Unwanted Biases with Adversarial Learning](https://api.semanticscholar.org/CorpusID:9424845). Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society.

The documentation and explanations are heavily based on the "[Mitigating Unwanted Biases in Word Embeddings with Adversarial Learning](https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb)" colab notebook.

Adversarial networks mitigate some biases in learned word representations based on the idea that predicting an outcome Y given an input X should ideally be independent of some protected variable Z. Informally, "knowing Y would not help you predict Z any better than chance." 

This can be achieved using two networks in a series, where the first attempts to predict Y using X as input, and the second attempts to use the predicted value of Y to predict Z. Please refer to Figure 1 of [Mitigating Unwanted Biases with Adversarial Learning](https://api.semanticscholar.org/CorpusID:9424845). Ideally, we would like the first network to predict Y without permitting the second network to predict Z any better than chance.
"""

"""
While we have input words X and learned word representations Y, it's not immediately clear what Z could be. We can construct our own Z by:
1) computing a bias direction (in this case, for binary gender)
2) computing the inner product of word embeddings and the bias direction

In this toy example, the task we will consider is analogy completion, i.e. predicting _ in A is to B, as C is to _. The prediction for D should not reveal binary gender information.
"""

embed, indices, words = load_vectors(client, analogies)

embed_dim = embed[0].flatten().size(0)
print("word embedding dimension: %d" % embed_dim)

pairs = [
      ("woman", "man"),
      ("her", "his"),
      ("she", "he"),
      ("aunt", "uncle"),
      ("niece", "nephew"),
      ("daughters", "sons"),
      ("mother", "father"),
      ("daughter", "son"),
      ("granddaughter", "grandson"),
      ("girl", "boy"),
      ("stepdaughter", "stepson"),
      ("mom", "dad"),
  ]

f = []
m = []
for wf, wm in pairs:
    f.append(embed[indices[wf]].reshape(1, -1))
    m.append(embed[indices[wm]].reshape(1, -1))
bias_direction = TwoMeansBiasDirection()(torch.cat(f), torch.cat(m))

word_vec = client.word_vec(WORD)
# print(WORD + ":", torch.dot(torch.tensor(word_vec), bias_direction).item())

# analogy_indices = filter_analogies(analogies, indices)
analogy_indices = analogies
data, labels, protect_labels = make_data(analogy_indices, embed, indices, bias_direction)
# print(data.size(), labels.size(), protect_labels.size())

"""
Training adversarial networks is extremely difficult. It is important to:
1) lower the step size of both the predictor and adversary to train both models slowly to avoid parameters diverging,
2) initialize the parameters of the adversary to be small to avoid the predictor overfitting against a sub-optimal adversary,
3) increase the adversary’s learning rate to prevent divergence if the predictor is too good at hiding the protected variable from the adversary
"""

# predictor = FeedforwardAnalogyPredictor(embed_dim, embed_dim, embed_dim)
predictor = SimpleAnalogyPredictor(embed_dim)
adversary = torch.nn.Linear(embed_dim, 1)

hyperparameters = {
    "predictor_lr": 2**-16,
    "adversary_lr": 2**-16,
    "adversary_loss_weight": 1.0,
    "num_epochs": 1000,
    "batch_size": 1000 
}

criterion = torch.nn.MSELoss()
predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=hyperparameters["predictor_lr"])
adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=hyperparameters["adversary_lr"])

A = "man"
B = "woman"
C = "boss"
NUM_ANALOGIES = 5

# Use a word embedding to compute an analogy
test_analogy = []
for i, word in enumerate((A, B, C)):
    test_analogy.append(torch.tensor(client.word_vec(word)).reshape(1, -1))
test_analogy = torch.cat(test_analogy).unsqueeze(0)

loader = torch.utils.data.DataLoader(torch.arange(data.size(0)), batch_size=hyperparameters["batch_size"])
for e in range(1, hyperparameters["num_epochs"] + 1):
    predictor.train()
    adversary.train()

    for idx in loader:
        predictor_optimizer.zero_grad()
        adversary_optimizer.zero_grad()

        # predict analogy
        pred = predictor(data[idx])
        protect_pred = adversary(pred)

        pred_loss = criterion(pred, labels[idx])
        protect_loss = criterion(protect_pred, protect_labels[idx])
        # print("pred_loss:", pred_loss.item())
        # print("protect_loss:", protect_loss.item())

        protect_loss.backward(retain_graph=True)
        protect_grad = {name: param.grad.clone() for name, param in predictor.named_parameters()}
        adversary_optimizer.step()

        predictor_optimizer.zero_grad()
        pred_loss.backward()

        with torch.no_grad():
            for name, param in predictor.named_parameters():
                unit_protect = protect_grad[name] / torch.linalg.norm(protect_grad[name])
                param.grad -= ((param.grad * unit_protect) * unit_protect).sum()
                param.grad -= hyperparameters["adversary_loss_weight"] * protect_grad[name]
        predictor_optimizer.step()
    
    predictor.eval()
    adversary.eval()

    with torch.no_grad():
        print("Epoch {}".format(e))
        # should predict "boss" and other non-gendered words with high probability
        pred = predictor(test_analogy).numpy()
        for neighbor, score in client.similar_by_vector(pred.flatten().astype(float), topn=5):
            print("%s : score=%f\n" % (neighbor, score))
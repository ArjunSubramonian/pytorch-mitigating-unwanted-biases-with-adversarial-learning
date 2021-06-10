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

# toy predictor
class FeedforwardAnalogyPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardAnalogyPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.output_size)
        self.projection = torch.nn.Linear(self.output_size, 1, bias=False)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        pred = self.fc2(relu)
        pred = -pred[:, 0, :] + pred[:, 1, :] + pred[:, 2, :]
        pred = pred - torch.matmul(self.projection(pred), self.projection.weight)
        return pred

class SimpleAnalogyPredictor(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAnalogyPredictor, self).__init__()
        self.hidden_size  = hidden_size
        self.projection = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, pred):
        pred = -pred[:, 0, :] + pred[:, 1, :] + pred[:, 2, :]
        pred = pred - torch.matmul(self.projection(pred), self.projection.weight)
        return pred

def load_analogies(filename):
  """Loads analogies.

  Args:
    filename: the file containing the analogies.

  Returns:
    A list containing the analogies.
  """
  analogies = []
  with open(filename, "r") as fast_file:
    for line in fast_file:
      line = line.strip()
      # in the analogy file, comments start with :
      if line[0] == ":":
        continue
      words = line.split()
      analogies.append(words)
  return analogies

def load_vectors(client, analogies):
  """Loads and returns analogies and embeddings.

  Args:
    client: the client to query.
    analogies: a list of analogies.

  Returns:
    A tuple with:
    - the embedding matrix itself
    - a dictionary mapping from strings to their corresponding indices
      in the embedding matrix
    - the list of words, in the order they are found in the embedding matrix
  """
  words_unfiltered = set()
  for analogy in analogies:
    words_unfiltered.update(analogy)

  vecs = []
  words = []
  index_map = {}
  for word in words_unfiltered:
    try:
      vecs.append(torch.nn.functional.normalize(torch.tensor(client.word_vec(word)).reshape(1, -1)))
      index_map[word] = len(words)
      words.append(word)
    except KeyError:
      print("word not found: %s" % word)

  return torch.cat(vecs), index_map, words

def make_data(
    analogies, embed, indices, 
    bias_direction):
  """Preps the training data.

  Args:
    analogies: a list of analogies
    embed: the embedding matrix
    indices: a dictionary mapping from strings to their corresponding indices
      in the embedding matrix
    bias_direction: the bias direction

  Returns:
    Three tensors corresponding respectively to the input, output, and
    protected variables.
  """
  data = []
  labels = []
  protect = []
  for analogy in analogies:
    # the input is just the word embeddings of the first three words
    data.append([])
    for w in analogy[:3]:
        data[-1].append(embed[indices[w]].reshape(1, -1))
    data[-1] = torch.cat(data[-1]).unsqueeze(0)
    # the output is just the word embeddings of the last word
    labels.append(embed[indices[analogy[3]]].reshape(1, -1))
    # the protected variable is the bias component of the output embedding.
    # the extra pair of [] is so that the tensor has the right shape after
    # it is converted.
    protect.append(torch.dot(embed[indices[analogy[3]]], bias_direction).reshape(1, -1))
  # Convert all three to tensors, and return them.
  return tuple(map(torch.cat, (data, labels, protect)))
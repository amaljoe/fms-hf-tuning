import random

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset

import random
from typing import List
import numpy as np
import math

class RLAgent:
    def __init__(
            self,
            weights: List[float],
            smoothing_factor: float = 0.9,
    ):
        self.num_domains = len(weights)
        self.weights = weights
        self._estimated_reward = [0] * self.num_domains
        total_weights = np.sum(weights)
        self._probabilities = [weight / total_weights for weight in weights]
        self.eps = 1 / self.num_domains
        self.prev_eps = None
        self.smoothing_factor = smoothing_factor
        self.vars_to_log = ["_probabilities", "_estimated_reward"]
        self.all_done = False
        self.iteration = 0

    def sample(self):
        index = random.choices(np.arange(self.num_domains), weights=self._probabilities)[0]
        return index

    def mark_done(self, index: int):
        """
        Marks the domain as done, which means it will not be sampled again.
        """
        self._probabilities[index] = 0
        total_weights = sum(self._probabilities)
        if total_weights > 0:
            self._probabilities = [p / total_weights for p in self._probabilities]
        else:
            self.all_done = True

    def update(self, index: int, reward: float) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """
        self.iteration += 1

        # update cumulative estimated reward
        self._estimated_reward[index] = self.smoothing_factor*self._estimated_reward[index] + (1-self.smoothing_factor)*math.exp(reward)

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_domains, math.sqrt(math.log(self.num_domains)/(self.num_domains*self.iteration)))

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._estimated_reward])
        scaling_factor = (1-self.num_domains*self.eps)/total_estimated_rewards

        # update weights
        for i in range(self.num_domains):
            self.weights[i] = math.exp(self._estimated_reward[i]*self.prev_eps)*scaling_factor + self.eps

        # update probabilities
        total_weights = sum(self.weights)
        for i in range(self.num_domains):
            self._probabilities[i] = self.weights[i]/total_weights
        return self._probabilities

    def reset(self):
        """
        Resets the agent's state.
        """
        self._estimated_reward = [0] * self.num_domains
        total_weights = np.sum(self.weights)
        self._probabilities = [weight / total_weights for weight in self.weights]
        self.eps = 1 / self.num_domains
        self.prev_eps = None
        self.all_done = False
        self.iteration = 0

    def group_update(self, idx: List[int], rewards: List):
        self.iteration += 1
        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_domains, math.sqrt(math.log(self.num_domains)/(self.num_domains*self.iteration)))

        # update cumulative estimated reward
        for index, reward in zip(idx, rewards):
            # smoothed mean
            # self._estimated_reward[name] = self.smoothing_factor*self._estimated_reward[name] + (1-self.smoothing_factor)*reward
            # smoothed exponentiated mean
            self._estimated_reward[index] = self.smoothing_factor*self._estimated_reward[index] + (1-self.smoothing_factor)*math.exp(reward)
        # print(f"Rank: {torch.distributed.get_rank()} -- estimated_reward {self._estimated_reward}")

        # calculate normalized scaling factor
        total_estimated_rewards = sum((r*self.prev_eps) for r in self._estimated_reward)
        scaling_factor = (1-self.num_domains*self.eps)/total_estimated_rewards

        # update weights
        for i in range(self.num_domains):
            # self.weights[self.dataset_map[name]] = math.exp(self._estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps
            self.weights[i] = self._estimated_reward[i]*self.prev_eps*scaling_factor + self.eps

        # update probabilities
        total_weights = sum(self.weights)
        for i in range(self.num_domains):
            self._probabilities[i] = self.weights[i]/total_weights

        return self._probabilities

# Non-Accelerated OnlineDataLoader
class  OnlineDataLoader(IterableDataset):
    def __init__(self, datasets, batch_size=1, shuffle=False, **kwargs):
        print("Initializing OnlineDataLoader")
        self.tokenizer = kwargs.pop('tokenizer', None)
        self.num_domains = len(datasets)
        self.num_samples = sum([len(dataset) for dataset in datasets])
        self.data_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs) for dataset in datasets]
        self.data_iterators = [iter(data_loader) for data_loader in self.data_loaders]
        self.rl_agent = RLAgent([1 for _ in range(self.num_domains)])

    def __iter__(self):
        print("Starting OnlineDataLoader iteration")
        while not self.rl_agent.all_done:
            if not self.data_iterators:
                break  # All iterators exhausted
            index = self.rl_agent.sample()
            try:
                batch = next(self.data_iterators[index])
            except StopIteration:
                # Remove exhausted iterator and skip this round
                self.rl_agent.mark_done(index)
                continue
            batch['metadata'] = {'domain_index': index}
            loss = 5 if index == 0 else 0

            # tokenize
            if self.tokenizer:
                if 'input' in batch:
                    batch['input_ids'] = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors='pt')['input_ids'][0]
                    batch['attention_mask'] = self.tokenizer(batch['input'], padding=True, truncation=True, return_tensors='pt')['attention_mask'][0]

            self.take_training_signals(batch, loss)
            yield batch

    def __len__(self):
        return sum(len(dl) for dl in self.data_loaders)

    def take_training_signals(self, batch, loss):
        print("Taking training signals")
        domain_index = batch['metadata']['domain_index']
        print("Sample: {}\tReward: {}\tProbabilities{}".format(batch, loss, self.rl_agent._probabilities))
        self.rl_agent.update(domain_index, reward=loss)

#
# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     from dataset_frameworks.simple_text_dataset import SimpleTextDataset
#
#     model_name = "meta-llama/Llama-3.2-3B"
#     texts1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
#     texts2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     dataloader = OnlineDataLoader([SimpleTextDataset(texts1, tokenizer), SimpleTextDataset(texts2, tokenizer)], batch_size=1)
#     for batch in dataloader:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
#         metadata = batch['metadata']
#         text = batch['text']
#         reward = 5 if text[0] in texts1 else 0
#         print("Sample: {}\tReward: {}\tProbabilities{}".format(text[0], reward, dataloader.rl_agent._probabilities))
#         dataloader.take_training_signals(batch, reward)

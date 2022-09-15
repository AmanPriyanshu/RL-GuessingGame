from env import Environment
from agent import RLAgent
import torch
import numpy as np
from tqdm import trange
import pandas as pd
from matplotlib import pyplot as plt

class Experiment:
	def __init__(self, num_agents, num_chits, majority_chits=None, seed=0):
		self.env = Environment(num_chits, majority_chits)
		self.num_agents = num_agents
		self.model = RLAgent(self.num_agents)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
		self.criterion = torch.nn.BCEWithLogitsLoss()
		self.seed = seed

	def run_single_episode(self):
		np.random.seed(self.seed)
		agent_orders = np.arange(self.num_agents)
		np.random.shuffle(agent_orders)
		agent_ids, order_nos, chit_receiveds, is_majority_acc_to_prev_agents, targets = [], [], [], [], []
		for i in range(self.num_agents):
			agent_id = agent_orders[i]
			order_no = i
			chit_received = self.env.pick_up()
			agent_ids.append(agent_id)
			order_nos.append([order_no])
			if len(chit_receiveds)<1 or np.mean(chit_receiveds)<0.5:
				is_majority_acc_to_prev_agents.append(-1)
			else:
				is_majority_acc_to_prev_agents.append(1)
			target = [1 if np.random.random()>=0.5 else 0, chit_received, 1-chit_received, 1 if np.mean(chit_receiveds)>=0.5 else 0]
			targets.append(target)
			chit_receiveds.append(chit_received)
		agent_ids = torch.tensor(agent_ids)
		order_nos = torch.tensor(order_nos)
		is_majority_acc_to_prev_agents = torch.tensor(is_majority_acc_to_prev_agents)
		targets = torch.tensor(targets).float()
		self.optimizer.zero_grad()
		out = self.model(agent_ids, is_majority_acc_to_prev_agents, order_nos)
		loss = self.criterion(out, targets)
		loss.backward()
		self.optimizer.step()
		pred = torch.argmax(out, 1)
		performance_per_agent = torch.tensor([target[i] for target,i in zip(targets, pred)])
		self.seed += 1
		return loss.item(), torch.mean(performance_per_agent).item(), performance_per_agent, agent_orders.tolist(), pred

	def run_experiment(self, num_episodes=100):
		bar = trange(num_episodes)
		history_performance, history_pred, history_in_game_order = [], [], []
		for episode_no in bar:
			loss, acc, performance_per_agent, agent_orders, pred = self.run_single_episode()
			order_performance = [performance_per_agent[agent_orders.index(i)] for i in range(self.num_agents)]
			order_pred = [pred[agent_orders.index(i)] for i in range(self.num_agents)]
			order_in_game_order = [agent_orders.index(i) for i in range(self.num_agents)]
			bar.set_description(str({"episode_no": episode_no+1, "loss": round(loss, 3), "acc": round(acc, 3)}))
			history_performance.append(order_performance)
			history_pred.append(order_pred)
			history_in_game_order.append(order_in_game_order)
		bar.close()
		self.plotter(history_pred, history_in_game_order, history_performance)
		history_in_game_order = pd.DataFrame(np.array(history_in_game_order), columns=["agent"+str(i) for i in range(self.num_agents)])
		history_performance = pd.DataFrame(np.array(history_performance), columns=["agent"+str(i) for i in range(self.num_agents)])
		history_pred = pd.DataFrame(np.array(history_pred), columns=["agent"+str(i) for i in range(self.num_agents)])
		df = pd.DataFrame({"agent"+str(i):[None, None] for i in range(self.num_agents)})

		history_in_game_order.to_csv("experiment.csv", index=False, mode="w")
		df.to_csv("experiment.csv", index=False, mode="a", header=False)
		history_performance.to_csv("experiment.csv", index=False, mode="a")
		df.to_csv("experiment.csv", index=False, mode="a", header=False)
		history_pred.to_csv("experiment.csv", index=False, mode="a")

	def plotter(self, history_pred, history_in_game_order, history_performance):
		# choice graph - the initials, the middlers, the latters
		f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 10))
		init_choices = [[p.item() for o,p in zip(in_game_order, pred) if o>=0 and o<self.num_agents//3] for in_game_order, pred in zip(history_in_game_order, history_pred)]
		init_choices_counts = np.zeros((len(init_choices), 4))
		for row_idx, choice in enumerate(init_choices):
			for i in range(4):
				count = len([c for c in choice if c==i])
				init_choices_counts[row_idx, i] = count
		ax1.set_title("Initially participating")
		for i, (c,l) in enumerate(zip(['r', 'g', 'b', 'y'], ["random", "picked up", "opposite of picked up", "majority"])):
			ax1.plot(init_choices_counts.T[i], c=c, label=l)

		init_choices = [[p.item() for o,p in zip(in_game_order, pred) if o>=self.num_agents//3 and o<2*(self.num_agents//3)] for in_game_order, pred in zip(history_in_game_order, history_pred)]
		init_choices_counts = np.zeros((len(init_choices), 4))
		for row_idx, choice in enumerate(init_choices):
			for i in range(4):
				count = len([c for c in choice if c==i])
				init_choices_counts[row_idx, i] = count
		ax2.set_title("Intermediately participating")
		for i, (c,l) in enumerate(zip(['r', 'g', 'b', 'y'], ["random", "picked up", "opposite of picked up", "majority"])):
			ax2.plot(init_choices_counts.T[i], c=c, label=l)

		init_choices = [[p.item() for o,p in zip(in_game_order, pred) if o>=2*(self.num_agents//3)] for in_game_order, pred in zip(history_in_game_order, history_pred)]
		init_choices_counts = np.zeros((len(init_choices), 4))
		for row_idx, choice in enumerate(init_choices):
			for i in range(4):
				count = len([c for c in choice if c==i])
				init_choices_counts[row_idx, i] = count
		ax3.set_title("Lastly participating")
		for i, (c,l) in enumerate(zip(['r', 'g', 'b', 'y'], ["random", "picked up", "opposite of picked up", "majority"])):
			ax3.plot(init_choices_counts.T[i], c=c, label=l)
		plt.legend()
		plt.show()

if __name__ == '__main__':
	exp = Experiment(20, 100, 52)
	exp.run_experiment()
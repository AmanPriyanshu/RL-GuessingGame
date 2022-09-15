import torch

class RLAgent(torch.nn.Module):
	def __init__(self, num_agents):
		super(RLAgent, self).__init__()
		self.agent_embeddings = torch.nn.Embedding(num_agents, 128)
		self.linear_1 = torch.nn.Linear(128, 64)
		self.linear_2 = torch.nn.Linear(64, 16)
		self.linear_3 = torch.nn.Linear(17, 4)
		self.activation = torch.nn.ReLU()

	def forward(self, agent_id, is_majority_acc_to_prev_agents, order_no):
		embeds = self.agent_embeddings(agent_id)
		signed_embeds = torch.mul(is_majority_acc_to_prev_agents, embeds.t()).t()
		out = self.linear_1(signed_embeds)
		out = self.activation(out)
		out = self.linear_2(out)
		out = self.activation(out)
		out = torch.cat((out, order_no), 1)
		out = self.linear_3(out)
		return out

if __name__ == '__main__':
	batch_x = {"agent_id": torch.tensor([0, 1, 2, 3]), "is_majority_acc_to_prev_agents": torch.tensor([-1, 1, -1, 1]), "order_no": torch.tensor([[12], [0], [76], [2]])}
	model = RLAgent(10)
	out = model(batch_x["agent_id"], batch_x["is_majority_acc_to_prev_agents"], batch_x["order_no"])
	print(out)
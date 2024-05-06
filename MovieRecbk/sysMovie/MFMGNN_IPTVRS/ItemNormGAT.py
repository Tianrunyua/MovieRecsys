import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree,dropout_adj
class ItemNormGAT(MessagePassing):
	def __init__(self, in_channels, out_channels,gat='inner product', normalize=True, bias=True, aggr='add', **kwargs):
		super(ItemNormGAT, self).__init__(aggr=aggr, **kwargs)
		self.aggr = aggr
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.gat = gat
		self.weight_cat = self.weight_gate = Parameter(torch.Tensor(out_channels*2, 1))

	def forward(self, x, edge_index, size=None):
		# pdb.set_trace()
		if size is None:
			edge_index, _ = remove_self_loops(edge_index)
			# edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		# pdb.set_trace()
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self,edge_index_i, x_i, x_j, size_i):

		if self.gat == 'inner product':
			self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
		elif self.gat == 'concat':
			self.alpha = torch.matmul(torch.cat((x_i, x_j), dim=1), self.weight_cat)

		self.alpha = softmax(self.alpha, edge_index_i, size_i)
		return x_j * self.alpha.view(-1, 1)





	def update(self, aggr_out):
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

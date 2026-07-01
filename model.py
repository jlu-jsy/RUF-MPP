import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    NNConv,
    Set2Set,
    global_mean_pool,
)
from torch_geometric.utils import degree


class GINConv(MessagePassing):
    """GIN convolution block used by the molecular graph encoder."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        return self.mlp((1 + self.eps) * x + out)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    """GCN convolution block with bond-feature encoding."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr="add")
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        root = F.relu(x + self.root_emb.weight) / deg.view(-1, 1)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + root

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class EdgeMPNNConv(nn.Module):
    """Edge-conditioned MPNN convolution based on BondEncoder and NNConv."""

    def __init__(self, emb_dim: int, hidden_dim: int = None, aggr: str = "add"):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim * emb_dim),
        )
        self.nnconv = NNConv(
            in_channels=emb_dim,
            out_channels=emb_dim,
            nn=self.edge_mlp,
            aggr=aggr,
        )
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.bond_encoder(edge_attr)
        out = self.nnconv(x, edge_index, edge_emb)
        return F.relu(self.bn(out))


class Molecular_Graph_Encoder(nn.Module):
    """Molecular graph encoder with atom encoding, degree encoding, and stacked GNN layers."""

    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float = 0.0,
        JK: str = "last",
        residual: bool = True,
        gnn_type: str = "gin",
    ):
        super().__init__()
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.emb_dim = emb_dim

        self.atom_encoder = AtomEncoder(emb_dim)
        self.centrality_emb = nn.Embedding(512, emb_dim)
        self.convs = nn.ModuleList()
        self.ln_list = nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == "gin":
                conv = GINConv(emb_dim)
            elif gnn_type == "gcn":
                conv = GCNConv(emb_dim)
            elif gnn_type == "mpnn":
                conv = EdgeMPNNConv(emb_dim)
            else:
                raise ValueError(f"Undefined GNN type: {gnn_type}")

            self.convs.append(conv)
            self.ln_list.append(nn.LayerNorm(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        row, _ = edge_index
        deg = degree(row, x.size(0), dtype=torch.long).clamp(max=510) + 1
        h = self.atom_encoder(x) + self.centrality_emb(deg)
        h_list = [h]

        for layer in range(self.num_layer):
            h_prev = h_list[layer]
            h_new = self.convs[layer](h_prev, edge_index, edge_attr)
            h_new = self.ln_list[layer](h_new)

            if layer != self.num_layer - 1:
                h_new = F.relu(h_new)
            h_new = F.dropout(h_new, self.drop_ratio, training=self.training)

            if self.residual:
                h_new = h_new + h_prev

            h_list.append(h_new)

        if self.JK == "last":
            return h_list[-1]
        if self.JK == "sum":
            return sum(h_list)
        raise ValueError("JK only supports 'last' or 'sum'.")


class CSFA(nn.Module):
    """Cross-Stitch Fusion Adapter for multi-task molecular property prediction."""

    def __init__(
        self,
        num_layer: int = 12,
        emb_dim: int = 512,
        num_tasks: int = 1,
        drop_ratio: float = 0.0,
        JK: str = "last",
        graph_pooling: str = "mean",
        gnn_type: str = "gin",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        self.gnn = Molecular_Graph_Encoder(
            num_layer=num_layer,
            emb_dim=emb_dim,
            drop_ratio=drop_ratio,
            JK=JK,
            residual=True,
            gnn_type=gnn_type,
        )

        if graph_pooling.startswith("set2set"):
            steps = 3 if "3" in graph_pooling else 2
            self.pool = Set2Set(emb_dim, processing_steps=steps)
            pool_out_dim = 2 * emb_dim
        elif graph_pooling in {"attn", "attention", "global_attn"}:
            gate_nn = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1),
            )
            self.pool = GlobalAttention(gate_nn=gate_nn)
            pool_out_dim = emb_dim
        else:
            self.pool = global_mean_pool
            pool_out_dim = emb_dim

        self.head_trunk = nn.Sequential(
            nn.Linear(pool_out_dim, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim),
        )

        self.task_head_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim // 2),
                    nn.ReLU(),
                )
                for _ in range(num_tasks)
            ]
        )
        head_in = emb_dim // 2

        self.use_cross_stitch = True
        self.cross_stitch_A = nn.Parameter(torch.eye(num_tasks))
        self.cross_stitch_softmax = True

        adapter_dim = emb_dim // 8
        self.task_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(head_in),
                    nn.Linear(head_in, adapter_dim),
                    nn.ReLU(),
                    nn.Linear(adapter_dim, head_in),
                )
                for _ in range(num_tasks)
            ]
        )

        for adapter in self.task_adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

        self.task_heads = nn.ModuleList([nn.Linear(head_in, 1) for _ in range(num_tasks)])
        self.return_attn = False
        self.task_weight_head = nn.Sequential(
            nn.Linear(pool_out_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_tasks),
        )

    def forward(self, data, return_emb: bool = False, emb_level: str = "graph"):
        node_repr = self.gnn(data.x, data.edge_index, data.edge_attr)
        graph_repr = self.pool(node_repr, data.batch)

        z = self.head_trunk(graph_repr)
        task_reprs = [mlp(z) for mlp in self.task_head_mlps]
        H = torch.stack(task_reprs, dim=1)

        if self.use_cross_stitch:
            A = self.cross_stitch_A
            if self.cross_stitch_softmax:
                A = torch.softmax(A, dim=1)
            H = torch.einsum("tk,bkd->btd", A, H)

        outputs = []
        head_embeddings = []
        for task_id, head in enumerate(self.task_heads):
            h = H[:, task_id, :]
            h_adapter = h + self.task_adapters[task_id](h)
            outputs.append(head(h_adapter))
            head_embeddings.append(h)

        pred = torch.cat(outputs, dim=1)

        emb_level = emb_level.lower()
        emb = graph_repr if emb_level == "graph" else torch.stack(head_embeddings, dim=1)

        if self.return_attn:
            attn_logits = self.task_weight_head(graph_repr)
            if return_emb:
                return pred, attn_logits, emb
            return pred, attn_logits

        if return_emb:
            return pred, emb
        return pred

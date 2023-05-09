import torch
from models.FLOT.flot.tools import ot
from models.FLOT.flot.models.graph import Graph
from models.FLOT.flot.models.gconv import SetConv
import time


class FLOT(torch.nn.Module):
    def __init__(self, nb_iter):
        """
        Construct a model that, once trained, estimate the scene flow between
        two point clouds.

        Parameters
        ----------
        nb_iter : int
            Number of iterations to unroll in the Sinkhorn algorithm.

        """

        super(FLOT, self).__init__()

        # Hand-chosen parameters. Define the number of channels.
        n = 32

        # OT parameters
        # Number of unrolled iterations in the Sinkhorn algorithm
        self.nb_iter = nb_iter
        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        # Entropic regularisation
        self.epsilon = torch.nn.Parameter(torch.zeros(1))

        # Feature extraction
        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n)

        # Refinement
        self.ref_conv1 = SetConv(3, n)
        self.ref_conv2 = SetConv(n, 2 * n)
        self.ref_conv3 = SetConv(2 * n, 4 * n)
        self.fc = torch.nn.Linear(4 * n, 3)

        self.total_time = 0.0

    def get_features(self, pcloud, nb_neighbors):
        """
        Compute deep features for each point of the input point cloud. These
        features are used to compute the transport cost matrix between two
        point clouds.
        
        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud of size B x N x 3
        nb_neighbors : int
            Number of nearest neighbors for each point.

        Returns
        -------
        x : torch.Tensor
            Deep features for each point. Size B x N x 128
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing list of nearest 
            neighbors (NN) and edge features (relative coordinates with NN).

        """

        graph = Graph.construct_graph(pcloud, nb_neighbors)
        x = self.feat_conv1(pcloud, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)

        return x, graph

    def refine(self, flow, graph):
        """
        Refine the input flow thanks to a residual network.

        Parameters
        ----------
        flow : torch.Tensor
            Input flow to refine. Size B x N x 3.
        graph : flot.models.Graph
            Graph build on the point cloud on which the flow is defined.

        Returns
        -------
        x : torch.Tensor
            Refined flow. Size B x N x 3.

        """
        x = self.ref_conv1(flow, graph)
        x = self.ref_conv2(x, graph)
        x = self.ref_conv3(x, graph)
        x = self.fc(x)

        return flow + x

    def forward(self, pclouds):
        """
        Estimate scene flow between two input point clouds.

        Parameters
        ----------
        pclouds : (torch.Tensor, torch.Tensor)
            List of input point clouds (pc1, pc2). pc1 has size B x N x 3.
            pc2 has size B x M x 3.

        Returns
        -------
        refined_flow : torch.Tensor
            Estimated scene flow of size B x N x 3.

        """
        # start_time = time.time()
        # Extract features
        feats_0, graph = self.get_features(pclouds[0], 32)
        feats_1, _ = self.get_features(pclouds[1], 32)

        # # Optimal transport
        # transport = ot.sinkhorn(
        #     feats_0,
        #     feats_1,
        #     pclouds[0],
        #     pclouds[1],
        #     epsilon=torch.exp(self.epsilon) + 0.03,
        #     gamma=torch.exp(self.gamma),
        #     max_iter=self.nb_iter,
        # )
        # row_sum = transport.sum(-1, keepdim=True)

        # # Estimate flow with transport plan
        # ot_flow = (transport @ pclouds[1]) / (row_sum + 1e-8) - pclouds[0]

        # # Flow refinement
        # refined_flow = self.refine(ot_flow, graph)
        # end_time = time.time()
        # self.total_time = self.total_time + end_time - start_time

        return pclouds[0], pclouds[1],  feats_0, feats_1

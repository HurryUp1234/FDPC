A fast density peaks clustering algorithm with sparse search
Author links open overlay panelXiao Xu a, Shifei Ding a b, Yanru Wang a, Lijuan Wang a, Weikuan Jia c
a
School of Computer Science and Technology, China University of Mining and Technology, Xuzhou 221116, China
b
Mine Digitization Engineering Research Center of Minstry of Education of the People's Republic of China, Xuzhou 221116, China
c
School of Information Science and Engineering, Shandong Normal University, Jinan 250358, China
Received 20 July 2019, Revised 23 November 2020, Accepted 27 November 2020, Available online 13 December 2020, Version of Record 31 December 2020.

What do these dates mean?

crossmark-logo
Show less
Add to Mendeley
Share
Cite
https://doi.org/10.1016/j.ins.2020.11.050
Get rights and content
Abstract
Given a large unlabeled set of complex data, how to efficiently and effectively group them into clusters remains a challenging problem. Density peaks clustering (DPC) algorithm is an emerging algorithm, which identifies cluster centers based on a decision graph. Without setting the number of cluster centers, DPC can effectively recognize the clusters. However, the similarity between every two data points must be calculated to construct a decision graph, which results in high computational complexity. To overcome this issue, we propose a fast sparse search density peaks clustering (FSDPC) algorithm to enhance the DPC, which constructs a decision graph with fewer similarity calculations to identify cluster centers quickly. In FSDPC, we design a novel sparse search strategy to measure the similarity between the nearest neighbors of each data points. Therefore, FSDPC can enhance the efficiency of the DPC while maintaining satisfactory results. We also propose a novel random third-party data point method to search the nearest neighbors, which introduces no additional parameters or high computational complexity. The experimental results on synthetic datasets and real-world datasets indicate that the proposed algorithm consistently outperforms the DPC and other state-of-the-art algorithms.

Access through your organization
Check access to the full text by signing in through your organization.

Introduction
Clustering is the most studied and widely applied method in unsupervised learning, revealing the intrinsic properties of data through the learning of unlabeled data points [1], [2], [3]. Clustering attempts to divide the points into several disjoint subsets so that the points in the same cluster are highly similar to one another and dissimilar to points in any other cluster [4], [5]. Many different clustering algorithms have been proposed, which can be roughly categorized into partition-based, grid-based, density-based, and hierarchical-based clustering [6], [7], [8], [9]. Significantly, the density-based clustering algorithms can discover the clusters regardless of their shape, and they can effectively find the number of clusters [10], [11], [12]. Recently, the density peaks clustering (DPC) algorithm has become a popular discussion topic due to its effectiveness and simple parameters [13]. The key step of the DPC algorithm is to identify cluster centers by drawing a decision graph based on the assumption that cluster centers have higher local-density and larger relative-separation characteristics [14]. After that, each non-center point is assigned to the same cluster as its nearest neighbor of higher local-density [15]. The DPC algorithm has become widely used in community discovery, biology, medicine, and other fields because of its robust performance [16], [17], [18], [19]. However, DPC has some shortcomings.

Recently, most works on DPC focus on improving clustering precision. Mehmood et al. [20] proposed a CFSFDP-HD algorithm to improve DPC. It estimates the local-density with a nonparametric density estimation method. The proposed algorithm is more robust and effective than DPC. Yang et al. [21] proposed a LPC algorithm, which defines a Laplacian centrality instead of the cluster peaks used in DPC to realize parameter-free clustering. Du et al. [22] introduced a DPC-KNN algorithm, which uses the concept of the KNN to redefine the local-density. The DPC-KNN takes the global distribution of the dataset into account and is more feasible and effective than DPC. Xie et al. [23] proposed a FKNN-DPC algorithm, which improves DPC through a novel local-density calculation method along with a novel allocation strategy. FKNN-DPC calculates the local-density based on the KNN and is independent of the cutoff distance.

Although DPC has reached a series of breakthroughs in clustering precision, in practical, DPC has relatively high computational complexity [24]. In DPC, a decision graph is constructed based on a similarity matrix to identify the cluster centers. As with other density-based algorithms, it requires calculating the similarity between each pair of points to create a similarity matrix. In order to improve the efficiency of DPC, a few methods have been proposed. These methods can be summarized into three categories: (1) Grid-based clustering: Xu et al. [25] proposed a DPCG algorithm, which introduces the idea of utilizing non-empty grids instead of all points according to the distribution of the datasets to reduce the distance calculations to enhance the DPC algorithm. Xu et al. [26] proposed two strategies named GDPC and CDPC, which select cluster centers by screening the effective partial points instead of all points. Both GDPC and CDPC can enhance the effectiveness of DPC since they apply the high local-density feature of the cluster centers. (2) Hybrid clustering: Bai et al. [27] combined the advantages of the k-means to improve the efficiency of DPC. The proposed CFSFDP + A and CFSFDP + DE are designed based on approximation concept and exemplar clustering to reduce distance calculations for rapid clustering. (3) Parallel clustering: Gong et al. [28] proposed an EDDPC algorithm applying MapReduce. It eliminates large scale excess distance calculations and the cost of data shuffling through the Voronoi diagram, data replication, and data filtering. Although the theoretical and practical advantages of the improvement algorithms mentioned above, they will introduce new parameters and sacrifice accuracy to reduce complexity.

In this paper, we propose an algorithm named the fast sparse search density peaks clustering (FSDPC) algorithm based on a sparse search strategy. The key idea behind FSDPC is that measuring the similarity between the nearest neighbors rather than all data points leads to more efficient clustering performance. Meanwhile, the similarity between the appropriate nearest neighbors will maintain a satisfactory clustering result. To search for the appropriate nearest neighbors, we made customized a novel random third-party data point method. To summarize, the significant contributions of the paper are as follows:
?
We propose a FSDPC algorithm to enhance the efficiency of DPC, which constructs a decision graph through fewer similarity calculations to reduce the computational complexity while maintaining a satisfactory clustering result.

?
We propose a novel sparse search strategy to construct an incomplete similarity matrix based on the nearest neighbors, which is inspired by the observed facts that the local-density and relative-separation calculation of one point only depends on the high similar nearest neighbors. We can rapidly obtain the ideal local-density and relative-separation for constructing a decision graph through this new similarity matrix.

?
We design a novel method to search for the appropriate nearest neighbors without introducing additional parameters and high computational complexity, which was never studied and addressed before. This new method is inspired by the idea: to measure the dissimilarity between two points, we only need to value their distance from a random third-party data point.

?
Our experimental results show that the FSDPC algorithm can maintain the same clustering precision as the DPC algorithm. In addition, based on the sparse search strategy, FSDPC can outperform DPC and several state-of-the-art analogous algorithms in terms of clustering efficiency.


The remaining parts of this paper are organized as follows: the DPC algorithm is reviewed in Section 2. Section 3 presents a FSDPC algorithm. Section 4 demonstrates the effectiveness and efficiency of FSDPC on both synthetic datasets and real-world datasets. Finally, the concluding generalization and further challenges are summarized.

Section snippets
Density peaks clustering algorithm
DPC is a popular density-based clustering algorithm that finds the number of clusters intuitively while discovering the clusters. It was introduced recently in Science, which has attracted great interest [29], [30], [31]. We analyze the theoretical underpinnings to understand the behaviors of the DPC algorithm.

Fast sparse search density peaks clustering algorithm
To overcome the high computational complexity of DPC, we proposed an algorithm named the fast sparse search density peaks clustering (FSDPC) algorithm. In FSDPC, a sparse search strategy is designed to construct a similarity matrix with fewer distance calculations, quickly obtain satisfactory clustering results.

Experimental analysis
In this section, we illustrate the effectiveness and efficiency of the FSDPC algorithm on eight synthetic datasets and eight real-world datasets. The scales of the datasets vary from small to large with the different numbers of clusters, as shown in Tables 1 and 2. The performance of FSDPC is compared with state-of-the-art improved algorithms, including GDPC and CDPC proposed in, [26] DPC-KNN proposed in, [22] and FKNN-DPC proposed in [23]. GDPC and CDPC are two algorithms for improving the

Conclusions
In this paper, we proposed a FSDPC algorithm, which reduces the computational complexity of the DPC and obtains as satisfactory clustering results as DPC. In FSDPC, a novel sparse search strategy is proposed for efficient clustering with fewer similarity calculations. FSDPC identifies accurate cluster centers based on the sparse search strategy rapidly. Furthermore, we have also proposed a random third-party data point method to avoid the high complexity and the additional parameters in the
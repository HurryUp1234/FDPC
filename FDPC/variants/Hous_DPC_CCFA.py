Density peak clustering based on relative density relationship
Author links open overlay panelJian Hou a, Aihua Zhang a, Naiming Qi b
Show more
Add to Mendeley
Share
Cite
https://doi.org/10.1016/j.patcog.2020.107554
Get rights and content
Highlights
?
We find out the reason behind two major problems afflicting the density peak clustering algorithm.

?
We introduce the concept of subordinate to describe the relative density relationship.

?
We propose a subordinate based criterion to solve the two problems.

?
Experiments and comparisons demonstrate the effect of our approach.


Abstract
The density peak clustering algorithm treats local density peaks as cluster centers, and groups non-center data points by assuming that one data point and its nearest higher-density neighbor are in the same cluster. While this algorithm is shown to be promising in some applications, its clustering results are found to be sensitive to density kernels, and large density differences across clusters tend to result in wrong cluster centers. In this paper we attribute these problems to the inconsistency between the assumption and implementation adopted in this algorithm. While the assumption is based totally on relative density relationship, this algorithm adopts absolute density as one criterion to identify cluster centers. This observation prompts us to present a cluster center identification criterion based only on relative density relationship. Specifically, we define the concept of subordinate to describe the relative density relationship, and use the number of subordinates as a criterion to identify cluster centers. Our approach makes use of only relative density relationship and is less influenced by density kernels and density differences across clusters. In addition, we discuss the problems of two existing density kernels, and present an average-distance based kernel. In data clustering experiments we validate the new criterion and density kernel respectively, and then test the whole algorithm and compare with some other clustering algorithms.

Access through your organization
Check access to the full text by signing in through your organization.

Introduction
Existing clustering algorithms can be categorized into a few types approximately. As a typical centroid-based clustering algorithm, the k-means algorithm is commonly used as a baseline in comparison. Spectral clustering [1], [2] performs dimension reduction and then groups data points in a lower dimensional space. In distribution-based clustering, the expectation maximization algorithms [3] use a mixture of distribution models to fit the data distribution of clusters. The affinity propagation (AP) [4], [5] algorithm passes affinity messages among data points to determine cluster centers and members gradually. Different from many partitioning-based algorithms, the dominant set (DSet) algorithm [6] defines a dominant set as a maximal subset of data points with internal coherency, and then detects dominant sets as clusters sequentially. Important works in clustering also include hierarchical clustering [7], mean shift [8], multi-view clustering [9], subspace clustering [10], [11], fuzzy c-means [12], correlation clustering [13], consensus clustering [14] and others [15].

In this paper our work falls into the type of density-based clustering, which has an attractive property of extracting clusters of spherical shapes (e.g., D31 [16]) and non-spherical shapes (e.g., Spiral [17]). The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm [18] uses a density threshold to detect cluster borders. Specifically, this algorithm uses a neighborhood radius Eps and the minimum number MinPts of points in the neighborhood to define a density threshold, based on which the core points, border points and noise points are defined and detected. As a generalization of DBSCAN, the OPTICS (Ordering Points To Identify the Clustering Structure) algorithm [19] removes the necessity of the parameter Eps and produces a hierarchical result. Many other works [20] in density-based clustering are also available in the literature.

While DBSCAN-like algorithms are based on a density threshold, the density peak clustering (DPC) algorithm [21] is presented based on two assumptions. The first assumption is that cluster centers are local density peaks, and the second one states that one data point is in the same cluster as its nearest higher-density neighbor. Based on the first assumption, this algorithm selects cluster centers as those data with both large local density 老 and large distance 汛 denoting the distance between one data point and its nearest higher-density neighbor. After that, the non-center data are grouped into respective clusters with the second assumption. While being simple, the DPC algorithm is shown to perform well in many experiments.

However, we have found that DPC clustering results are influenced by density kernels, and large density differences across clusters may degrade clustering results. We perform an in-depth investigation of this algorithm and find that these problems are resulted by the 老 criterion in cluster center identification. This algorithm is proposed based on two assumptions, and both assumptions are based on the relative density relationship among data. However, the 老 criterion in cluster center identification reflects absolute density value, but not relative density relationship. In our opinion, this inconsistence between the assumption and implementation results in the problems of DPC in clustering experiments. In order to remove this inconsistency, we define subordinate as a concept to describe the relative density relationship and present an alternative criterion to replace 老. Our new criterion utilizes only relative density relationship, and is therefore less influenced by density kernels and density differences across clusters. In addition, we discuss the problems of the cutoff kernel and the Gaussian kernel and propose to overcome the problems with a new density kernel. On the basis of the preliminary version [22], our attributions in this paper are as follows. We find that the problem of the DPC algorithm lies in the inconsistency between the assumption and implementation, thereby pointing out a new direction to improve this algorithm. We define the concepts of direct subordinate and indirect subordinate, and study their effects in detecting cluster centers. Furthermore, we use the number of direct subordinates in nearest neighbors to refine the criterion with respect to the one in the preliminary version [22]. In addition, we discuss the robustness to data perturbation, generalization performance and the computation complexity of our algorithm.

Some other works have also identified the limitations of the DPC algorithm and presented different solutions. In [23] the authors point out the problems of the original DPC algorithm, including the definition of local density and distance measure, the one-step allocation strategy and the cutoff distance dc which is difficult to determine. A shared-nearest-neighbor-based clustering method is then proposed to tackle these problems. A different approach is presented in [24], where a nonparametric method is used to estimate probability distribution based on heat diffusion, accounting for both selection of the cutoff distance and boundary correction of the kernel density estimation. In order to overcome the defects from the dc parameter, [25] introduces the idea of K-nearest neighbors to compute dc and the local density, and applies a new approach to select initial cluster centers automatically. Noticing that the original DPC algorithm tends to ignore the cluster centers in sparse regions, [26] redefines the local density of a point as the number of points whose neighbors contain this point, thereby treating the centers in sparse and dense regions equally. Here we observe that all these works lay more or less importance on designing new local density estimation methods in improving the original DPC algorithm. This is reasonable as DPC is a density-based clustering algorithm. Different from these works, in this paper our importance is not on designing new density kernels, as the local density is not used directly in our approach. Instead, we make use of only relative density relationship to present an alternative criterion to 老 in cluster center identification. This new criterion removes the inconsistency between the assumption and implementation of the DPC algorithm, and therefore is able to solve the problems caused by the inconsistency.

In Section 2 we introduce how the DPC algorithm works and illustrate its problems. The concept of subordinate, the new criterion and density kernel are derived in Section 3. In Section 4 we conduct experiments to test the new criterion and density kernel separately, and also compare the whole algorithm with other approaches. Section 5 summarizes the final conclusions.

Section snippets
Density peak clustering algorithm
In this section we present the idea and basic clustering process of the density peak algorithm, and then illustrate its problems with examples.

Our algorithm
From Section 2.2 we observe that both the two assumptions of the DPC algorithm are based only on relative density relationship. However, in implementation this algorithm uses absolute density value 老 as one cluster center identification criterion. In our opinion, this inconsistency between the assumption and implementation results in some problems in clustering experiments. This observation prompts us to explore an alternative criterion of 老 to make use of only relative density relationship.

Experiments
Our major work in this paper consists of a density kernel and a criterion of cluster center identification. In experiments we first test the effects of the proposed kernel and criterion separately. Then we compare the whole algorithm with some other commonly used algorithms. Finally, we study the robustness performance to data perturbation and compare the running time between our algorithm and DP with different density kernels.

In total 28 datasets are used in experiments, including 12 synthetic

Conclusions
In this paper we present a cluster center identification criterion based only on relative density relationship to enhance the density peak clustering algorithm. We study the density peak clustering algorithm and attribute its problems to the inconsistency between its assumption and implementation. In order to remove the inconsistency, we present the concept of subordinate to describe the relative density relationship, and use the number of subordinates as a criterion in cluster center
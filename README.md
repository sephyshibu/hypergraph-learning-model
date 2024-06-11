# hypergraph-learning-model
The project compares two feature selection methods: a hypergraph learning model and an association score method with the p-value. A hypergraph learning model is a novel approach that aims to capture high-order interactions among features, while the association score method is a traditional method that measures the strength of the association between each feature and the target variable.The project uses the diabetes dataset to train the SVM models and evaluates the performance of both feature selection methods based on the accuracy in predicting diabetes. The accuracy of each approach is calculated and compared, with the hypergraph learning method achieving higher accuracy than the association score method.The study emphasises how crucial it is tochoose the best feature selection techniques in order to increase the precision of prediction models. The comparison of the hypergraph learning and association score methods provides insights into the effectiveness of these methods for predicting diabetes using the SVM algorithm


**Methodolgy**
The goal of this project was to compare the performance of two distinct feature selection methods when using the SVM algorithm to predict diabetes. The first method of feature selection is through the use of a hypergraph learning model. This method involves measuring high-order correlations between features using hypergraph theory. By identifying and analyz- ing these correlations, an optimal subset of features. Another method involves assigning an association score to each feature and then determining the statistical significance of that scoreusing a p-value.Both of these methods are valuable tools for feature selection and can help to improved the accuracy and efficiency of machine learning models. The dataset used in the study consisted of 32,000 instances, 8 characteristics, and a target variable indicating the presence or absence of diabetes. During the feature selection phase, a hypergraph learning model was built to demonstrate the high-order connections among the extracted features, which were subsequently clustered. The PCA method was then used to reduce the dimensionality of each cluster before utilizing the SVM-RFE feature ranking approach to generate a ranking of the remaining features in these clusters. Finally, an optimal feature subset was created by repeatedly selecting and validating features, taking into consideration the high-order relations in the clusters and the ranking outcome. This optimal feature subset was then used to classify diabetics using the SVM algorithm. The following section will provide a detailed description of each step of the process.

**Feature Extraction**
During the feature extraction process, texture features are obtained using a method called texture analysis. A total of 32,000 features are extracted using this method. These texture features can provide valuable information about the structure and composition dataset. Therefore, it’s crucial to carefully select the most relevant features to ensure the best possible outcomes.

**Feature Selection**
In this project, the effectiveness of two different feature selection methods for predicting diabetes using the SVM algorithm was compared. The first method involved using a hypergraph learning model for feature selection, while the second method used an association score with p-value to select features.


**High order correlation measurement**
The combination of covariance and feature ranking method has been used for diagnosing diabetes disease, resulting in an excellent classification performance. The covariance only measures pairwise correlation and is unable to assess high-order correlation across numerous fea- tures, even though adopting feature correlation in feature selection can result in a better feature subset. The proposed approach HOCDF (High-Order Correlation Detecting in Features) splits the feature selection procedure into two parts to address this restriction. All features are first clustered to measure their high-order correlation, and then feature clusters are used in the second phase to create the ideal feature subset. Suggested clustering method groups features with strong correlations into the same cluster and those with weak correlations into multiple clusters in order to capture high-order correlations between features. The technique clusters the features using shared entropy, hypergraph theory, and community learning by graph approximation (CLGA).

**Shared entropy**
First, a method to assess this correlation is required in order to find cluster features based on the high-order correlation between features. The shared entropy is employed as the measurement technique in this research. The joint entropy, shown in Eq. (3.1), can be used to calculate the amount of information included in a sets of Y1, Y2,..., Ym (m ≥ 2).

J(Y 1,Y 2...Y m) =m∑y1,y2...ym
P(Y 1,Y 2...Y m) log2 P(y1, y2...ym) (3.1)
Where P(y1, y2,...ym) it defines the joint probability of m features. The definition of shared
entropy is based on the joint entropy, and it is expressed in Eq.(3.2):
S(Y 1,Y 2...Y m) = (−1)0 ∑i = 1mJi + (−1)1 ∑1 ≤ i ≤ j ≤ mJi j + .... + (−1)m−1J1, 2...m

Where Ji is an abbreviation of J (Yi), Jij is an abbreviation of J(Yi, Yj), and J1,2,...m is abbreviated for the joint entropy J(Y1, Y2,...Ym) of m features.
The level of correlation among features increases as the shared entropy value increases. However, it is important to note that only nominal features can be measured using shared entropy.Therefore, continuous features must be discretized before they can be analyzed. To do this, all features are first normalized using the Min-Max normalization technique to ensure they fall within the range of [0,1]. The range [0,1] is then divided into 100 equal intervals, and each feature is discretized to its nearest point. This process ensures that the continuous features are transformed into nominal features that can be measured using shared entropy.

**Hypergraph learning model**

Hypergraphs differ from classic graphs in that edges are defined differently. In classic graphs, each edge connects two vertices, but in hypergraphs, an edge or hyperedge can connect any number of vertices.The shared entropy is a metric for assessing the links between features, and hypergraphs can express high-order correlations among numerous vertices or features. A vertex can be used to represent each feature, and a hypergraph can be created to displaythe connections between the feature vertices. Several parameters need to be determined before constructing the hypergraph, the two parameters that determine the hypergraph’s scales are ∆d, the highest degree of hyperedges, and ∆ s, the low shared entropy in the hypergraph, which controls the scales of the hypergraphs. Additionally, a parent-child relationship can be definedbetween two hyperedges as PC(ei, ej). If ei and ej satisfy PC(ei,ej)= 1, then ei is the parent
hyperedge of ej, and conversely, ej is the child hyperedge of ei.If a hyperedge’s shared entropy is greater than one, it will be formed. The set of all hy-
peredges with a degree of one is denoted as E1 =e1, e2,...ek k ≤ m. All 2-degree (d(e) = 2)hyperedges are produced in the second iteration. For each pair of features, the shared entropy Sij is determined using the formula Sij = S(Yi, Yj)0 ≤i ≤j ≤ m. If Sij s, then the connection between feature vertex Yi and Yj is high enough to warrant the construction of a hyperedge with the shared entropy Sij serving as the weight Wij. A hyperedge set E2 then displays the second-
level hyperedges. The final step of the second iteration is the removal of all parent hyperedges from E1 of these hyperedges in E2.
After being built, the hypergraph can be represented as G = V, E, W, where V is the set of all features vertices, E is the set of all hyperedges, and W is the set of all hyperedge weights, which represent the correlation’s strength. In this work, shared entropy is used to express the weight. In addition, the relationship between the point and the edge is represented by an incidence matrix H ( |V | x |E|). Below is a list of H’s definitions.
H(v,e)=1, v ∈ e
0, otherwise

The diagonal matrix W denoted as (|E| × |E|) contains all the weights of hyperedges. Using
this, we can define the adjacent matrix A denoted as (|V | × |V |) as
A=HWHT
26
Here, HT (|E| x |v|) is denoted as the transpose of H. The matrix A represents the relationship
between each pair of features.
CLGA
The next step is to cluster the features by decomposing the high-dimensional feature set into
low-dimensional feature clusters once the correlation between the features has been represented
by a hypergraph. The initial setting of a parameter establishes the necessary number of clusters.
The CLGA method is then applied to the matrix A that was previously calculated, producing
a feature clusters set C (c1, c2..., c) and a partition membership matrix B (|V | × |C|). Which
cluster a feature belongs to is indicated by this matrix B.
B(v,c)=1, v ∈ c
0, otherwise

**Optimal Features Subset generations**

It is crucial to fully utilise this correlation after discovering high-order correlation between
characteristics and clustering them properly. To achieve this, it is necessary to use a feature
selection approach to extract the best subset of characteristics from these clusters. To create a
precise and effective model, the feature selection process involves choosing a subset of pertinent
features. The accuracy of the model can be increased via feature selection by lowering the
amount of pointless or redundant features. In addition to taking into account the connection
between features and choosing features that are significantly connected with the target variable,
an efficient feature selection method should be able to extract the most insightful and pertinent
features from the clusters. Overall, choosing the right features can increase model performance,
cut down on computational complexity, and make models easier to understand.

**SVM-RFE with Feature Cluster**

The next step is to create the ideal feature subset from the feature clusters after measuring
the high-order correlation between the features. This methodology consists of three steps. All
of the features in the final best or optimal subset are first produced from the feature clusters,
which are then used as the method’s input. The feature dimensionality reduction technique is
put into place for each feature cluster in the second step. Reducing the number of identical
features is the aim of this approach because having too many might lead to redundancy and
waste of computing resources. As a result, the number of features in each cluster is decreased
using the PCA approach. .
To construct the ideal feature subset, the method is the Sequential Forward Selection in the
third stage. The SFS approach chooses one feature at a time and slowly adds it to the current
feature set, starting with an empty feature set. The selected feature either shares a cluster with
the top-ranked feature or has the highest ranking among the features that were not selected.
Prior to selecting K features, the K features in the same clusters as the feature with the highest
rating are chosen. After corrections have been made, the performance of the feature subset in
classification is once more assessed by performing an SVM cross-validation experiment on a
sample set and tracking the average accuracy.
When there are no longer any features in any cluster, the process is finished because no more
distinct feature subsets can be produced. The most accurate feature subset, which is also the
best feature subset, is the method’s ultimate result.

**Feature Selection to use p-Value**

A statistical notion utilised in hypothesis testing is the P-Value. It is the minimal probability
value deemed significant for a certain test statistic. The observed test statistic’s significance in
respect to the null hypothesis is assessed using the P-Value. The value of alpha =0.05 is used in
a hypothesis test to identify properties that are significant for the class.
The null hypothesis (Ho) is accepted in this hypothesis test if the P-Value is smaller than
alpha, indicating that the characteristic significantly affects the class. The null hypothesis (Ho)
is rejected, however, if the P-Value is bigger than alpha proving that the characteristic does not
significantly affect the class. It’s vital to remember that alpha is a constant that denotes the
degree of significance that the researcher chooses to use in reaching a conclusion.
Therefore, the P-Value is a crucial concept in statistical hypothesis testing as it helps in mak-
ing decisions about whether to reject or accept the null hypothesis. The null hypothesis repre-
sents the absence of a relationship or effect between variables, while the alternative hypothesis
suggests the presence of a relationship or effect. By conducting hypothesis testing and deter-
mining the P-Value, researchers can make more informed decisions about the significance of
their findings.
One model for determining the link between the category response variable and one or more
continuous predictor variables or categories is the support vector machine. There are two cat-
egories in the answer variable: ”presence of diabetes” (y = 1) and ”absence of diabetes” (y =
0).

**Classification**

After obtaining the best feature subset, the following stage involves incorporating it with a
classifier for detecting Diabetes. For this purpose, Support Vector Machine (SVM) is employed
as the classifier in this project. SVM is widely used in supervised learning in classification
tasks that identifies the optimal hyperplane for separating distinct classes in data. In detecting
Diabetes, the SVM classifier utilizes the chosen features as input and trains a model capable of
distinguishing between individuals with and without Diabetes with high accuracy.
Through the use of a cross-validation experiment on a sample set, the SVM classifier’s per-
formance is assessed. To do this, separate the data into several subsets, train the classifier using
part of the subsets, then test it using the remaining subset. To guarantee that the performance
measures are accurate and unbiased, the procedure is performed several times, with various sub-
sets being utilised for testing and training each time. The number of right predictions the model
makes and the actual labels of the data are compared to determine the classifier’s accuracy.
Overall, the process of integrating the optimal feature subset with the SVM classifier involves
training the classifier on the selected features and testing its performance using cross-validation.
This allows us to evaluate the accuracy of the classifier and determine its effectiveness in de-
tecting Diabetes based on the selected features

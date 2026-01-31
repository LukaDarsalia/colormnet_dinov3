## ColorMNet: A Memory-based Deep

## Spatial-Temporal Feature Propagation Network

## for Video Colorization

```
Yixin Yang, Jiangxin Dong, Jinhui Tang, and Jinshan Pan
```
```
Nanjing University of Science and Technology
```
(^30101) Running time (/s) 100
31
32
33
34
35
36
PSNR (dB)
DeepExemplar
BiSTNet
Color2Embed
TCVC (^) DDColor
ColorMNet (Ours)
VCGAN
DeepRemaster
(a) Input and exemplar (b) DeepRemaster [9]
(c) DeepExemplar [39] (d) ColorMNet (Ours) (e) Performance and running time comparison
Fig. 1:Colorization results on a real-world video and model performance comparisons
between our proposed ColorMNet and other methods on the DAVIS [25] dataset in
terms of PSNR and running time. State-of-the-art methods [9, 39] do not generate
well-colorized images in (b) and (c). In contrast, by exploring the features from large-
pretrained visual models to estimate robust spatial features for each frame, effectively
propagating these features along the temporal dimension based on memory mechanisms
for far-apart frames, and exploiting the video property that adjacent frames contain
similar contents, our method accurately restores the colors on the grass and generates
a realistic image in (d). (e) shows that the proposed ColorMNet performs favorably
against state-of-the-art methods in terms of accuracy and running time. The size of
the test images for measuring the running time is 960 × 536 pixels.
Abstract.How to effectively explore spatial-temporal features is im-
portant for video colorization. Instead of stacking multiple frames along
the temporal dimension or recurrently propagating estimated features
that will accumulate errors or cannot explore information from far-apart
frames, we develop a memory-based feature propagation module that
can establish reliable connections with features from far-apart frames
and alleviate the influence of inaccurately estimated features. To ex-
tract better features from each frame for the above-mentioned feature
propagation, we explore the features from large-pretrained visual mod-
els to guide the feature estimation of each frame so that the estimated
features can model complex scenarios. In addition, we note that adja-
cent frames usually contain similar contents. To explore this property
for better spatial and temporal feature utilization, we develop a local
attention module to aggregate the features from adjacent frames in a
spatial-temporal neighborhood. We formulate our memory-based feature

# arXiv:2404.06251v1 [cs.CV] 9 Apr 2024


2 Y. Yanget al.

```
propagation module, large-pretrained visual model guided feature esti-
mation module, and local attention module into an end-to-end trainable
network (named ColorMNet) and show that it performs favorably against
state-of-the-art methods on both the benchmark datasets and real-world
scenarios. The source code and pre-trained models will be available at
https://github.com/yyang181/colormnet.
```
```
Keywords:Exemplar-based video colorization·Deep convolutional neu-
ral network·Feature propagation
```
## 1 Introduction

Due to the technical limitations of old imaging devices, lots of videos captured
in the last century are in black and white, making them less visually appealing
on modern display devices. As most of these videos have historical values and
are difficult to reproduce, it is of great need to colorize them.
Restoring high-quality colorized videos is challenging as it not only needs to
handle the colorization of each frame but also requires exploring temporal in-
formation from the video sequences. Therefore, directly applying existing image
colorization methods [3,10,12,16,19,29,35,40,42] does not generate satisfactory
colorized videos as minor perturbations in consecutive input video frames may
lead to substantial differences in colorized video results. To overcome this, nu-
merous methods model the temporal information from inter-frames by stacking
multiple frames along the temporal dimension [1, 9] or recurrently propagating
features [17,32,33,39]. Although these approaches show better performance than
the ones based on single image colorization, stacking multiple frames along the
temporal dimension cannot effectively leverage spatial-temporal prior from adja-
cent frames and requires a large amount of GPU memory. In addition, recurrent-
based feature propagation is not able to effectively explore long-range informa-
tion, leading to unsatisfactory results for frames far apart.
To better explore long-range temporal information, several approaches [20,36]
develop bidirectional recurrent-based feature propagation methods for video col-
orization. As the recurrent-based feature propagation treats the features of each
frame equally, if the features are not estimated accurately, the errors will accu-
mulate, thus affecting the final video colorization. Therefore, it is still challenging
to effectively model temporal information from long-range frames.
In addition to the temporal information exploration, how to extract good
features from each frame plays a significant role in video colorization. Existing
methods [1, 17, 32, 36, 39, 44] usually utilize a pretrained VGG [13] or ResNet-
101 [6] to extract features from each frame. These methods are able to model local
structures but are less effective for exploiting non-local and semantic structures,
e.g., complex scenes with multiple objects. To restore high-quality videos, it is
of great interest to develop a better feature representation method that is able
to characterize the non-local and semantic properties of each frame.
In this paper, we present a memory-based deep spatial-temporal feature prop-
agation network for video colorization. Note that the robustness of the spa-


```
ColorMNet: A MSTFP Network for Video Colorization 3
```
tial features extracted from each frame is important, we first develop a large-
pretrained visual model guided feature estimation (PVGFE) module, which is
motivated by the success of large-pretrained visual models [22] in generating ro-
bust visual features to facilitate the spatial feature estimation. However, simply
recurrently propagating the estimated spatial features or directly stacking them
along the temporal dimension does not effectively explore temporal information
for video colorization. Moreover, it requires a large amount of GPU memory
capacity to store past frame representations when the spatial resolution is large
and videos are long. To overcome these problems, we then propose a memory-
based feature propagation (MFP) module that can not only adaptively explore
and propagate useful features from far-apart frames but also reduce memory
consumption. In addition, we note that adjacent frames of a video usually con-
tain similar contents and thus develop a local attention (LA) module to better
utilize the spatial and temporal features. Taken together, the memory-based
deep spatial-temporal feature propagation network, called ColorMNet, is able to
generate high-quality video colorization results (see Figure 1(e)).
The main contributions are summarized as follows:

- We propose a large-pretrained visual model guided feature estimation mod-
    ule to model non-local and semantic structures of each frame for colorization.
- We develop a memory-based feature propagation module to adaptively ex-
    plore temporal features from far-apart frames and reduce memory usage.
- We develop a local attention module to explore similar contents of adjacent
    frames for better video colorization.
- We formulate the proposed network into an end-to-end trainable framework
    and show that it performs favorably against state-of-the-art methods on both
    the benchmark datasets and real-world scenarios.

## 2 Related Work

User-guided image colorization.Since the colorization problem is ill-posed,
conventional image colorization methods usually adopt local user hints [2,7,19,
21, 26, 27, 37, 42, 43] to make this problem well-posed. However, these methods
do not fully exploit the property of video sequences and usually need to solve
temporal consistency problems. In addition, their colorization performance for
individual frame is usually far from satisfactory as estimating global and seman-
tic features is challenging.
Automatic video colorization.Instead of using user-guided image methods,
several approaches explore deep learning to solve video colorization. In [17,44],
both the colorization performance and the temporal consistency are enhanced
by recurrently propagating features from adjacent frames for video colorization.
In [20], Liuet al. use bidirectional propagation to explore temporal features
and introduces a self-regularization learning scheme to minimize the prediction
difference obtained with different time steps. Although using feature propagation
improves the temporal consistency, it is not a trivial task to estimate the spatial
features from each frame due to the inherent complexity of scenes in videos.
Moreover, these automatic video colorization methods [24,28,35] may work well
on synthetic datasets but lack generality on diverse real-world scenarios.


4 Y. Yanget al.

```
PVGFE 𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝐂 Softmax Decoder Channel concatenation
```
```
𝟑𝟑×𝟑𝟑
```
```
Colorized video frames
```
```
MFP
```
**Exemplar**

```
Softmax
```
```
Input Key features Value features
```
```
LA
```
```
Output
```
```
 Frozen weights
Feature map
Element-wise summation
```
```
Matrix product
CSpatial concatenation
Softmax
 DINOv
```
```
 ResNet
```
```
𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝐂
```
```
𝟑𝟑×𝟑𝟑
```
```
𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝐂
```
```
𝟑𝟑×𝟑𝟑
```
```
𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝐂
```
```
𝟑𝟑×𝟑𝟑
```
```
Embedded feature
```
```
PVGFE
```
```
Colorized video frames and exemplar image
Channel split
ResNet
𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝟑𝟑×𝟑𝟑 𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝐂𝟑𝟑×𝟑𝟑
```
```
Keep every -th frame
```
```
Select features based on cumulative similarities
```
```
(b)
```
```
C C
```
```
Temporally
```
```
Spatially
```
```
(c)
```
```
(a)
```
Fig. 2:An overview of the proposed ColorMNet. The core components of our method
include:(a)large-pretrained visual model guided feature estimation (PVGFE) module,
(b)memory-based feature propagation (MFP) module and(c)local attention (LA).

Exemplar-based video colorization.Exemplar-based methods aim to gener-
ate videos that are faithful to the exemplar with enhanced temporal consistency.
In [39], Zhanget al. employ a recurrent-based feature propagation module to
explore temporal features for latent frame restoration. In [9], Iizukaet al. pro-
pose stacking multiple frames along the temporal dimension to obtain better
performance. To better explore spatial and temporal features, Chenet al. [1]
adopt ResNets [6] instead of the commonly-used VGG [13] model for better spa-
tial feature estimation and split video sequences into frame blocks for long-term
spatiotemporal dependency. In [36], Yanget al. use bidirectional propagation to
gradually propagate features and use optical flow models [31] to align adjacent
frames. Recurrently propagating information from long-range frames or stack-
ing multiple frames improves the colorization performance. However, if there are
inaccurately estimated features of long-range frames, the errors will be accumu-
lated, which thus affects video colorization. Additionally, when dealing with long
videos, these methods often consume significant GPU memory and suffer from
slow inference speeds, posing substantial challenges to video colorization.

## 3 ColorMNet

Our goal is to develop an effective and efficient video colorization method to
restore high-quality videos with low GPU memory requirements. The proposed
ColorMNet contains a large-pretrained visual model guided feature estimation
(PVGFE) module to extract spatial features from each frame, a memory-based
feature propagation (MFP) module that is able to adaptively explore the tem-
poral features from far-apart frames, and a local attention (LA) module that is
used to explore the similar contents from adjacent frames for better spatial and
temporal feature utilization. Figure 2 illustrates the overview of the proposed
method. In the following, we explain each module in detail.


```
ColorMNet: A MSTFP Network for Video Colorization 5
```
3.1 PVGFE module
To estimate robust spatial features, we explore the features learned from large-
pretrained visual models as they are able to model the non-local and semantic
information and are robust to numerous scenarios. Note that one of the large-
pretrained visual models,i.e., DINOv2 [22], adopts ViT [4] as the main feature
extractor and generates all-purpose visual features facilitating both image-level
(classification) and pixel level (segmentation) tasks due to its robust feature
representation ability. In this paper, we utilize the global features learned from
DINOv2 to guide the local features learned from CNNs for better feature esti-
mation of each frame.
Given the input grayscale frames{Xi}Ni=1(whereXi∈RH×W×^1 ,H×W
denotes the spatial resolution,Nis the number of frames of the input video), we
first extract features{Gi}Ni=1and{Li}Ni=1by applying the pretrained DINOv
and ResNet50 [6] to{Xi}Ni=1, respectively. Then we use the cross attention [38]
to fuse{Gi}Ni=1and{Li}Ni=1for obtaining robust spatial features.
Specifically, we first extract the query featureQGi fromGi, the key feature
KiLand the value featureViLfromLiby:

```
QGi = Conv 3 × 3 (Gi), (1a)
KLi = Conv 3 × 3 (Li), (1b)
ViL= Conv 3 × 3 (Li), (1c)
```
where{QGi,KLi,ViL}∈R
Hˆ×Wˆ×Cˆ
,Hˆ×Wˆ andCˆdenote the spatial and channel
dimensions, respectively;Conv 3 × 3 (·)denotes a convolution with the filter size of
3 × 3 pixels. We denote the matrix forms ofQGi,KLi,ViLasQˆGi,KˆLi,VˆLi, and
obtain the fused feature by:

```
ˆFi= softmax
```
### (

```
QˆGi(KˆLi)⊤
α
```
### )

```
VˆLi, (2)
```
whereQˆGi ∈R
Cˆ×HˆWˆ
,KˆLi ∈R
Cˆ×HˆWˆ
andVˆLi ∈ R
Cˆ×HˆWˆ
are obtained by
reshaping tensors [38] from the original sizeRHˆ×Wˆ×Cˆ;softmax(·)denotes the
softmax operation that is applied to each row of the matrix;αis a scaling factor.
In our implementation, we take the features of the last 4 layers of ViT-
S/14 from DINOv2 and concatenate them together in channel dimension as the
featureGi. For the featureLi, we take stage-4 features with stride 16 from
the base ResNet50 [6]. Finally, the featureˆFi ∈ RCˆ×HˆWˆ is reshaped into

Fi ∈RHˆ×Wˆ×Cˆfor the following processing.

3.2 MFP module
Inspired by the efficient mechanisms of human brains in memorizing long-term
information,i.e., paying more attention to frequently used information, we pro-
pose a memory-based feature propagation module to effectively and efficiently
explore temporal features by establishing reliable connections with features from
far-apart frames and alleviating the influence of inaccurately estimated features.
Assuming that we have predictedi− 1 color frames{I 1 ,···,Ii− 1 }with their
chrominance channels{Y 1 ,···,Yi− 1 }and luminance channels{X 1 ,···,Xi− 1 }


6 Y. Yanget al.

(i.e., the input grayscale frames), we aim to estimate the chrominance channel
Yiof thei-th color frameIibased on{Y 1 ,···,Yi− 1 }and the given exemplarR.
First, we extract features{F 1 ,F 2 ,···,Fi}andFrfrom{X 1 ,X 2 ,···,Xi}and
the luminance channelXrofRusing the proposed PVGFE module for global
and semantic features in the spatial dimension. Meanwhile, we extract features
{E 1 ,···,Ei− 1 }andErfrom{Y 1 ,···,Yi− 1 }and the chrominance channelYrof
Rusing a lightweight pretrained ResNet18 [6]. Then, we generate the embedded
query, key, and value features by:

QFi = Conv 3 × 3 (Fi), (3a)
{KjF}ij−=1^1 ={Conv 3 × 3 (Fj)}ij−=1^1 , (3b)
KrF= Conv 3 × 3 (Fr), (3c)
{VjE}ij−=1^1 ={Conv 3 × 3 (Ej)}ij−=1^1 , (3d)
VrE= Conv 3 × 3 (Er). (3e)
As{K 1 F,···,KiF− 1 }and{V 1 E,···,ViE− 1 }contain valuable historical infor-
mation of previously colorized video frames{I 1 ,···,Ii− 1 }, they facilitate the
establishment of long-range temporal correspondences between the contents of
the current frame and those of the previously colorized video frames. However,
memorizing all of the historical frames results in significant GPU memory con-
sumption, especially as the number of colorized frames increases. As a trade-off
between long-range correspondence and memory consumption, we keep everyγ
frame and discard the remaining ones that contain similar contents to obtain
more temporally compact and representative features.
In particular, given that adjacent frames are mutually redundant, we first
merge the temporal features by concatenating everyγframe, thereby estab-
lishing reliable connections with far-apart frames under constrained memory
consumption, and obtain the aggregated key featuresAK 1 by:

```
AK 1 = Concat(KFγ,KF 2 γ,···,KFzγ), (4)
```
where{KFγ,KF 2 γ,···,KFzγ}are the matrix forms of{KγF,K 2 Fγ,···,KFzγ},z=

⌊i−γ^1 ⌋,⌊·⌋denotes the round down operation, andConcat(·)denotes the spatial
dimension concatenation operation.
Note that errors are likely to accumulate when the features of some frames are
not estimated accurately. In addition, the high dimension (HˆWˆ) ofAK 1 makes it
impossible to handle lots of video frames with limited GPU memory capacity. To
overcome these problems, we then aggregateAK 1 into a more spatially compact
and less error-prone form by selecting better features with higher usage. However,
the computation of the feature usage frequency relies on a sufficient number of
colorized frames. Therefore, when the number of colorized frames is small (i.e.,
z < Ns), we directly useAK 2 as the output of our proposed MFP module, which
is defined as:
AK 2 = Concat

### (

```
KFr,AK 1
```
### )

### , (5)

whereKFr is the matrix form ofKFr. When a sufficient number of frames is
colorized (i.e.,z=Ns), we aggregateAK 1 by compressing the earlierNeframes
to reduce GPU memory consumption and alleviate the influence of inaccurately


```
ColorMNet: A MSTFP Network for Video Colorization 7
```
estimated features for better video colorization. After the aggregation, the total
number of aggregated frames reduced fromz=Nstoz=Ns−Neand we use
AK 2 in (5) as the output of the MFP module untilzreachesNsagain. In the
following, we explain the situation whenz=Nsin detail.

We first define the cumulative similarities{Sγ,···,Szγ}for the key features
{KFγ,···,KFzγ}, which are aggregated in (4), as:

```
Sγ=
```
```
i∑− 1
```
```
j=γ+
```
### H∑ˆWˆ

```
m=
```
### (

```
softmax
```
### (

```
−Cj
```
### ))

```
,Cj= (Cjm,n), (6a)
```
```
Cjm,n=
```
### ∥

```
∥KFj(:,m)−KFγ(:,n)
```
### ∥

### ∥^2

2 , (6b)
whereSγ∈R^1 ×HˆWˆ andCj∈RHˆWˆ×HˆWˆ,KFj(:,m)denotes them-th feature

vector inKFj andKFγ(:,n)denotes then-th feature vector inKFγ,{KFj,KFγ}∈

RCˆ

k×HˆWˆ
,∥·∥ 2 denotes the Euclidean distance calculation. Then, we normalize
Sγby dividing it by the number of frames in{KFj}ij−=^1 γ+1for fairness consider-

ation and obtainS

```
′
γ=
```
Sγ
i− 1 −γ, which can be utilized to estimate the probability
that the featureKFγ is accurately predicted asS

′
γdescribes how frequently the
feature is used. We further define a top-MoperationTM(·)to select the bestM
pixels of all features from the earlierNeframes{KFγ,···,KFNeγ}, based on the

highestMvalues in{S
′
γ,···,S

```
′
Neγ}as:
Kc= Concat
```
### (

```
KFγ,···,KFNeγ
```
### )

```
, (7a)
```
```
Sc= Concat
```
### (

### S

```
′
γ,···,S
```
```
′
Neγ
```
### )

```
, (7b)
```
```
TM(Kc) = Concat (Kc(:,1),···,Kc(:,M)), (7c)
```
whereKc∈RCˆ
k×NeHˆWˆ
,Sc∈R^1 ×NeHˆWˆ, and{Kc(:,1),···,Kc(:,M)}denote
theMfeature vectors inKcthat satisfy{Sc(:,1),···,Sc(:,M)}are the top-M
values inSc. Then we obtain the aggregated key featuresAK 2 by:

```
AK 2 = Concat
```
### (

```
TM(Kc),KFr,KF(Ne+1)γ,···,KFzγ
```
### )

### , (8)

whereAK 2 ∈R
Cˆk×T
,T=M+ (1 +z−Ne)HˆWˆ. Similarly, we obtain the aggre-

gated value featuresAV 2 ∈R
Cˆv×T
by aggregating{VrE,{VjE}ij−=1^1 }.
To fully exploit the temporal information contained inAK 2 andAV 2 , we use
a commonly usedL 2 similarity to find the features inAK 2 that are most similar
toQFi, whereQFi ∈RCˆ
k×HˆWˆ
is the matrix form ofQFi, by:
Wi= softmax

### (

```
−Di
```
### )

```
,Di= (Dim,n), (9a)
```
```
Dim,n=
```
### ∥

```
∥QFi(:,m)−AK 2 (:,n)
```
### ∥

### ∥^2

```
2 , (9b)
```
where{Wi,Di} ∈RHˆWˆ×T,QFi(:,m)denotes them-th feature vector inQFi
andAK 2 (:,n)denotes then-th feature vector inAK 2. Then, we reconstruct the

i-th estimated featureVi∈R
Cˆv×HˆWˆ
of the chrominance channel by mapping
the aggregated value featuresAV 2 based onWias:

```
Vi=AV 2 (Wi)⊤. (10)
```

8 Y. Yanget al.

Finally, we obtain the estimated featureViby reshapingVito its original size
R
Hˆ×Wˆ×Cˆv

. In the following, we enhanceViby a local attention (LA) module.

3.3 LA module

As adjacent frames contain similar contents that may be useful to complement
the long-range information captured by MFP, we develop a local attention (LA)
module to explore better spatial-temporal features.
We first useQpi∈R^1 ×^1 ×
Cˆk
to represent the featureQFi in (3a) at the spatial

locationp∈R
Hˆ×Wˆ

. Next, we formulate the pastdkey features in (3b) and past
dvalue features in (3d) of the spatial-temporal neighborhood corresponding to
Qpias:

```
KN(p),c= Concat(KiN−(dp),···,KNi−( 1 p)), (11a)
VN(p),c= Concat(ViN−d(p),···,ViN−( 1 p)), (11b)
```
whereKjN(p)andVjN(p)denote thej-th key feature and value feature corre-

sponding to aλ×λpatchN(p)centered atp,KN(p),c∈Rd×λ×λ×
Cˆk
andVN(p),c
∈Rd×λ×λ×Cˆ
v

. Then we apply the local attention toQpiwithKN(p),candVN(p),c
and obtain the output of LA module at locationpas:

```
Lpi= softmax
```
### (

```
Qpi(KN(p),c)
```
```
⊤
```
```
β
```
### )

```
VN(p),c, (12)
```
whereQpi ∈R^1 ×Cˆ

```
k
,KN(p),c∈Rdλ
```
(^2) ×Cˆk
andVN(p),c∈Rdλ
(^2) ×Cˆv
are matrices
obtained by reshapingQpi,KN(p),candVN(p),cfrom their original size,βis the
scaling factor. Finally, we obtain the featureLi∈RHˆ×Wˆ×Cˆ
v
by reshapingLito
its original size.
To restore the colors of the input frameXi, we further adopt a simple but
effective decoder. Specifically, we use a decoderD(·)consisting of ResBlocks [6]
followed by up-sampling interpolation layers to gradually refine the enhanced
featureVi+Liand obtain the predictedi-th chrominance channelYias:
Yi=D(Vi+Li). (13)

## 4 Experimental Results

In this section, we first describe the experimental settings of the proposed Col-
orMNet. Then we evaluate the effectiveness of our approach against state-of-the-
art methods. More experimental results are included in the supplemental mate-
rial. Code and models are available at https://github.com/yyang181/colormnet.

4.1 Experimental settings

Datasets.Following previous works [1,20,36], we use the datasets of DAVIS [25]
and Videvo [15] for training and generate grayscale video frames using OpenCV
library. For testing, we use three popular benchmark test datasets including the
DAVIS validation set, the Videvo validation set and the official validation set of
NTIRE 2023 Video Colorization Challenge [11] (NVCC2023 for short).


```
ColorMNet: A MSTFP Network for Video Colorization 9
```
Table 1:Quantitative comparisons of the proposed method against state-of-the-art
ones on the DAVIS [25] validation set (short frame length), the Videvo [15] validation
set (medium frame length) and the NVCC2023 [11] validation set (long frame length).

Our method achieves favorable performance in most of the metrics. Top (^1) stand (^2) nd
results are marked inbold redand blue respectively.∗denotes that we apply DVP [18]
to the results of Color2Embed and DDColor as post-processing method. Note that for
the LPIPS matrix on Videvo, we examine the performance of our method and BiSTNet
in additional decimal places to determine the best one and the second best one as the
performance of these two methods appears to be identical when displayed in the table
with a limited number of decimal places.†denotes that two exemplars are used.
Methods DDColor[12] Color2Embed[43] DDColor
∗
[12]
Color2Embed∗
[43]
VCGAN
[44]
TCVC
[20]
DeepExemplar
[39]
DeepRemaster
[9]
BiSTNet†
[36]
ColorMNet
(Ours)
Categories Image-based Fully-automatic Exemplar-based
DAVIS
PSNR (dB)↑ 30.84 31.33 30.67 31.05 30.24 31.10 33.24 33.25 34.02 35.
FID↓ 65.13 101.08 86.96 118.11 128.48 116.41 69.56 92.28 44.69 38.
SSIM↑ 0.926 0.951 0.936 0.943 0.924 0.955 0.950 0.961 0.964 0.
LPIPS↓ 0.085 0.076 0.086 0.086 0.100 0.080 0.062 0.060 0.043 0.
Videvo
PSNR (dB)↑ 30.76 31.65 30.60 31.62 30.62 31.29 33.11 32.95 34.12 34.
FID↓ 45.08 66.73 50.80 73.02 97.86 80.74 54.93 68.15 32.25 30.
SSIM↑ 0.925 0.958 0.940 0.960 0.934 0.956 0.956 0.964 0.968 0.
LPIPS↓ 0.079 0.060 0.077 0.061 0.085 0.068 0.053 0.053 0.036 0.
NVCC
PSNR (dB)↑ 29.95 30.90 30.16 30.66 29.77 30.45 32.03 32.25 33.18 33.
FID↓ 48.50 65.92 95.83 70.35 86.59 72.76 35.39 53.03 25.55 20.
SSIM↑ 0.888 0.933 0.911 0.927 0.866 0.935 0.930 0.951 0.949 0.
LPIPS↓ 0.114 0.091 0.099 0.098 0.130 0.089 0.073 0.071 0.054 0.
Exemplars.Following [1, 9, 36, 39], we adopt a similar strategy to utilize the
first frame of each video clip as an exemplar to colorize the video clip.
Evaluation metrics.Following the experimental protocol of most existing col-
orization methods, we use peak signal-to-noise ratio (PSNR), structural similar-
ity index measurement (SSIM) [34], the Fréchet Inception Distance (FID) [8],
and the learned perceptual image patch similarity (LPIPS) [41] as evaluation
metrics. These assessment matrices cover a spectrum of pixel-wise considera-
tions, the distribution similarity between generated images as well as ground
truth images, and the perception similarity.
Implementation details.We train our model on a machine with one RTX
A6000 GPU. We adopt the Adam optimizer [14] with default parameters using
PyTorch [23] for 160,000 iterations. The batch size is set to 4. We adopt the CIE
LAB color space for each frame in our experiments. The learning rate is set to
a constant 2 × 10 −^5. We empirically setγ= 5,Ne= 5,Ns= 10,M= 128and
d= 1. We employ anL 1 loss, computed as the mean absolute errors between
the predicted images and the ground truths.
4.2 Comparisons with state-of-the-art methods
Quantitative comparison.We benchmark our method against state-of-the-art
ones on three datasets and report quantitative results. The competing methods
include the automatic colorization techniques [20,44], single exemplar-based ap-
proaches [9, 39], and a double exemplar-based method [36]. Furthermore, for
enhanced self-containment, we incorporate comparisons with leading image col-
orization methods [12,43] and enhance the temporal consistency of these methods
by applying DVP [18] as post-processing method. For [9,20,36,44], whose weights


10 Y. Yanget al.

```
(a) (b) (c) (d)
```
Input frame and exemplar image (e) (f) (g) (h)
Fig. 3:Qualitative comparisons on clipparkourfrom the validation set of DAVIS [25]
dataset. (a)-(g) are the colorization results by DDColor [12], TCVC [20], VCGAN [44],
DeepRemaster [9], DeepExemplar [39], BiSTNet†[36] and ColorMNet (Ours). (h)
Ground truth. The evaluated methods do not generate realistic colorful images in
(a)-(f). In contrast, our approach generates a well-colorized image in (g).

are trained on the same datasets as our method, we conduct tests using their
official codes and weights provided by the authors. However, for [12,39,43], we
train from scratch using the official codes on identical datasets as our method
to ensure fair comparisons. In addition, we adhere to the commonly adopted
protocol of using the same first frame ground truth image of each video clip
as the exemplar for testing all exemplar-based methods [9, 36, 39, 43], with the
exception that we also include the last frame as an extra exemplar for testing
the double exemplar-based method, BiSTNet [36].

Table 1 shows that the ColorMNet consistently generates competitive col-
orization results on the DAVIS [25] validation set, the Videvo [15] validation set,
and the NVCC2023 [11] validation set, where our method performs better than
the evaluated methods in terms of PSNR, SSIM, FID, and LPIPS, indicating
that our method not only can generate both high-quality and high-fidelity col-
orization results, but also demonstrates the good generalization based on the
favorable performance on the validation set in NVCC2023 (the training set in
NVCC2023 is not included in the training phase).

Qualitative evaluation.Figure 3(a) shows that the image-based method [12]
generates results with non-uniform colors on the man’s face and cloth. Automatic
video colorization techniques [20,44] can not generate vivid colors (Figure 3(b)
and (c)). Exemplar-based methods [9, 36, 39] can not establish long-range cor-
respondence and thus fail to restore the colors on the man’s arms and cloth
(Figure 3(d)-(f)). In contrast, the proposed ColorMNet generates a vivid col-
orized image (Figure 3(g)) by modeling both the spatial information from each
frame and the temporal information from far-apart frames.

Evaluations on real-world videos.We further evaluate the proposed method
on a real-world grayscale video,Manhattan (1979). We obtain the exemplars
(Figure 4(b)) by searching the internet to find the most visually similar images
to the input video frames. Figure 4(c) and (d) show that state-of-the-art meth-
ods [9, 39] do not colorize the objects (e.g., the wall of the building, the trees
and the sky) well. In contrast, our method generates better-colorized frames,
where the colors look natural and realistic (Figure 4(e)). In addition, our method


```
ColorMNet: A MSTFP Network for Video Colorization 11
```
(a) (b) (c) (d) (e)
Fig. 4:Qualitative colorization compar-
isons on real-world video Manhattan
(1979). (a) Input frame. (b) Exemplar im-
ages obtained by Google Image Search.
(c)-(e) are the colorization results by
DeepRemaster [9], DeepExemplar [39]
and ColorMNet (Ours), respectively. The
methods [9,39] do not colorize the wall of
the building, the trees, and the sky well
in (c) and (d). Our ColorMNet generates
error-free and realistic colors in (e).

```
Table 2:Quantitative evaluations of the
video colorization methods with better
accuracy performance on the DAVIS [25]
dataset in terms of maximum GPU mem-
ory consumption, average running time
and temporal consistency index CDC.
Methods Memory (G) Running time (/s) CDC↓
DeepExemplar [39] 19.0 0.80 0.
DeepRemaster [9] 16.6 0.61 0.
BiSTNet [36] 34.9 1.62 0.
ColorMNet (Ours) 1.9 0.07 0.
Table 3:Effectiveness of the proposed
PVGFE, MFP, and LA modules in our
ColorMNet. Evaluated on the DAVIS [25].
ComponentsMethods ResNet50 DINOv2 PVGFEFeature extractor Stacking Recurrent MFPFeature propagationLocalityLAPSNRMetrics↑SSIM↑
ColorMNetColorMNetw/ ResNet50w/ DINOv2!! !! !! 35.01 0.96235.38 0.
ColorMNetColorMNetw/ Concatenationw/ Stacking !!!!!!! !! 35.26 0.96533.94 0.
ColorMNetColorMNetw/ Recurrent !!!!! 35.26 0.
ColorMNet (Ours)w/o LA !!!!!! !!! 35.77 0.97035.44 0.
```
demonstrates its robustness by consistently generating similar results even when
provided with exemplar images that possess diverse colors and contents.
Efficiency evaluation.Given that practical applications of video colorization
often involve processing longer videos, where maximum GPU memory usage and
inference speed are critical metrics, we further evaluate our method against three
representative state-of-the-art exemplar-based video colorization approaches [9,
36,39]. Specifically, we record the maximum GPU memory consumption during
the inference on a machine with an NVIDIA RTX A6000 GPU. The average
running time is obtained using 300 test images with a 960 × 536 resolution.
Table 2 shows that the maximum GPU consumption of ColorMNet (ours)
is only 11.2% of DeepRemaster [9], 10.0% of DeepExemplar [39] and 5.4% of
BiSTNet [36]; the running time is at least 8×faster than the evaluated methods.
Temporal consistency evaluations.To examine whether the colorized videos
generated by our method have a better temporal consistency property, we use
the color distribution consistency index (CDC) [20] as the metric. Table 2 shows
that our method has a lower CDC value when compared to exemplar-based
methods [9, 36, 39] on the DAVIS [25] validation set, which indicates that our
method is capable of generating videos with improved temporal consistency by
exploring better temporal information.

## 5 Analysis and Discussion

To better understand how our method solves video colorization and demonstrate
the effectiveness of its main components, we conduct a deeper analysis of the
proposed approach. For the ablation studies in this section, we train our method
and all alternative baselines on the training set of the DAVIS [25] dataset and
the Videvo [15] dataset with 160,000 iterations for fair comparisons.
Effectiveness of PVGFE.The proposed PVGFE explores robust spatial fea-
tures that can model both global semantic structures and local details for better
video colorization. To demonstrate its effectiveness, we compare with baseline
methods that respectively replace the PVGFE with the pretrained ResNet50 [6]


```
12 Y. Yanget al.
```
```
(a) (b) (c)
```
```
Input frame and exemplar image (d) (e) (f)
Fig. 5: Effectiveness of PVGFE for video colorization. (a) Input patch. (b)-
(e) are the colorization results by ColorMNetw/ ResNet50, ColorMNetw/ DINOv2,
ColorMNetw/ Concatenationand ColorMNet (Ours), respectively. (f) Ground truth. Com-
pared to the baselines, our approach yields a more natural colorized result in (e).
```
```
(a) (b) (c) (d) (e) (a) (b) (c) (d) (e)
Fig. 6:Visualization of features. We use the PCA tools by [22]. (a) Input frame.
(b)-(e) are the features generated by the feature extractors of ColorMNetw/ ResNet50,
ColorMNetw/ DINOv2, ColorMNetw/ Concatenationand ColorMNet (Ours), respectively.
Compared with (b), (c) and (d), our proposed PVGFE can generate features that are
not only semantic-aware (i.e., the players and the dancer in the foreground) but also
sensitive to local details (i.e., a crowd of spectators in the background) in (e).
```
35.3535.

35.4535.

35.5535.

35.6535.

35.7535.
35.
13579111315171921 PSNR (dB) vs 2325272931333537394143454749

```
35.70 4 6 8 10 12
```
```
35.
```
```
35.
```
```
35.
```
```
35.
```
```
PSNR (dB)
```
```
Ne=2Ne=3 PSNR vs Ns vs Ne
Ne=4Ne=5Ne=
Ne=7Ne=8Ne=
Ne=10Ne=
```
```
Ne
```
```
r = 5, M = 128
```
```
35.766535.
```
```
35.767535.
```
```
35.768535.
```
```
35.769535.
```
```
35.770535.
```
```
35.771535.
```
```
326496 PSNR (dB) 128160192 vs 224 M 256288320
```
```
(a) Ablation on differentγ (b) Ablation onNeandNs (c) Ablation on differentM
Fig. 7:Extensive ablation study on the detailed design of the proposed MFP module.
(ColorMNetw/ ResNet50for short), the pretrained DINOv2 [22] (ColorMNetw/ DINOv
for short), and the concatenation of both pretrained ResNet50 and DINOv
(ColorMNetw/ Concatenationfor short) in our implementation. Table 3 shows that
our ColorMNet with the PVGFE outperforms all baseline methods. The qualita-
tive comparisons in Figure 5 show that the results obtained by baseline methods
exhibit severe color distortions on the cloth of the player (Figure 5(b)-(d)). In
contrast, our proposed ColorMNet with the PVGFE generates a better-colorized
frame in Figure 5(e). Note that the PVGFE can adaptively enhance useful
features while reducing the influence of useless information based on the sim-
ilarities computed on input features by employing cross-attention (i.e., (2)).
However, a direct concatenation of features evenly without discrimination,i.e.,
ColorMNetw/ Concatenation, is less effective for reducing the impact of useless fea-
tures, thus degrading performance in Table 3 and Figure 5(d).
To better understand the feature estimators mentioned above, we use the
PCA tools by [22] to visualize the features generated by them. Figure 6(b) shows
that ResNet50 cannot generate features that are aware of semantic structures.
Although DINOv2 generates more semantic features in Figure 6(c) and (d), it
lacks the local details vital for colorization tasks, which explains why DINOv
```

```
ColorMNet: A MSTFP Network for Video Colorization 13
```
performs favorably in high-level vision tasks,i.e., classification and segmentation
(see [22] for details), but fails in colorization as the exact colors for pixels of
objects are crucial considerations in colorization, unlike in segmentation where
the primary decision is whether or not a pixel belongs to a human. Figure 6(e)
shows that our proposed PVGFE module is capable of generating better features
optimized for colorization, retaining both semantic relevance and local details.

Effectiveness of MFP.The proposed MFP propagates temporal features for
better long-range correspondences. To investigate whether directly stacking mul-
tiple frames along the temporal dimension or recurrently propagating features
can already generate competitive results, we compare with baseline methods
that respectively replace the MFP with direct stacking of the features from all
previous colorized frames and the exemplar image along the temporal dimen-
sion (ColorMNetw/ Stackingfor short) and the recurrent-based feature propaga-
tion [17,32,33,39] (ColorMNetw/ Recurrentfor short) in our implementation.

Table 3 shows that the PSNR value of our ColorMNet is at least 0. 51 dB
higher than each baseline method, which illustrates the effectiveness of the pro-
posed MFP in propagating features for video colorization. Figure 8(b) shows
that the baseline that directly stacks frames is not able to generate a realistic
image as spatial-temporal priors are not well-explored. The result obtained by
the baseline with recurrent-based feature propagation contains significant color
distortions on the boy (Figure 8(c)), as the errors accumulate in the recurrent-
based propagation steps. In contrast, the proposed ColorMNet using the MFP
generates a vivid and error-free image in Figure 8(d).
We further conduct an extensive ablation study on the parameters of the
proposed MFP module. Figure 7(a) shows that our method achieves its peak
PSNR whenγequals 5, as a higherγrisks potential information loss, while
a lowerγcould contribute redundant data. Figure 7(b) and (c) show that our
method generally achieves slightly higher PSNR values withNe,NsandM
increasing, respectively. However, note that the GPU memory usage escalates
correspondingly with larger values ofNe,NsandM.

Efficiency of MFP.To examine the efficiency of the proposed MFP, we fur-
ther evaluate the proposed ColorMNet against the baseline method with direct
stacking (i.e., ColorMNetw/ Stacking) on the validation set of NVCC2023 [11]
in terms of the maximum GPU memory consumption and the average running
time. Table 4 shows that our ColorMNet only requires 7 .8%of the maximum
GPU consumption of the baseline method, but the average running time of our
approach is nearly 14×faster than the baseline method, which indicates the
efficiency of the proposed MFP.
Tables 3 and 4 show that our approach using the MFP achieves a favorable
performance in terms of faster inference speed, lower GPU memory consump-
tion, and better colorization results, which demonstrates the effectiveness and
efficiency of the proposed MFP in video colorization.

Effectiveness of LA.To demonstrate the effect of the proposed LA, we further
compare with a baseline method that removes the LA module (ColorMNetw/o LA
for short) in our implementation. Table 3 shows that our ColorMNet using the


14 Y. Yanget al.

Table 4:Efficiency of the proposed MFP
module for video colorization.
Methods Memory consumption (G) Running time (/s)
ColorMNetw/ Stacking 24.1 1.
ColorMNet (Ours) 1.9 0.

(a) (b) (c) (d) (e)
Fig. 8:Effectiveness of the MFP mod-
ule for video colorization. (a) Input frame
and exemplar image. (b)-(d) are the col-
orization results by ColorMNetw/ Stacking,
ColorMNetw/ Recurrent and ColorMNet
(Ours), respectively. (e) Ground truth.
Compared with (b) and (c), the colorized
result (d) by our method contains fewer
color distortions and more vivid details.

```
(a) (b) (c) (d)
Fig. 9: Effectiveness of the proposed
LA module for video colorization. (a)
Input frame and exemplar image. (b)
and (c) are the colorization results by
ColorMNetw/o LAand ColorMNet (Ours).
(d) Ground truth. Our approach is able to
generate better-colorized result in (c).
,'` ． 忒－.』1. `b
```
```
•·
(a) Inputs (b)MAMBA[30] (c)MeMOTR[5] (d) Ours
Fig. 10:Comparison results with closely-
related methods on the DAVIS [25].
```
LA generates better results with higher PSNR and SSIM values than the baseline
method. Figure 9(b) shows that the baseline method without the LA does not
exploit the prior information among consecutive frames and thus cannot restore
the colors on the sky and the leaves. However, our approach generates vivid and
realistic colors in Figure 9(c), which demonstrates that the proposed LA module
is effective in capturing and leveraging better spatial-temporal features.
Closely-related methods.To the best of our knowledge, we are the first to
optimize a memory bank strategy suitable for colorization, yet it should be
acknowledged that related strategies have been explored in some video process-
ing works,e.g., MAMBA [30] constructs a memory bank to solve video object
detection by employing random selection strategy, MeMOTR [5] introduces a
long-term memory to solve video object tracking by assigning exponentially
decaying weights to it. Unlike MAMBA which applies a randomized selection
approach, treating every feature on par, or MeMOTR which updates past mem-
orized features via exponentially decaying weights, our proposed MFP module
stores features based on their importance which is determined by the frequency
of usage, thus empowering the ability of global relation mining.
We further adopt the random selection in MAMBA and the decaying weights
in MeMOTR to replace our MFP for comparison. To ensure a fair comparison,
the same training settings are kept for model testing. Figure 10 shows that our
method can generate better colors for the dancing girl and the green grass.
Limitations.Our proposed method aims to further enhance video colorization
performance while reducing GPU memory usage. However, there are limitations
in the model complexity,i.e., our model requires 123.61 Million parameters.

## 6 Conclusion

We present an effective memory-based deep spatial-temporal feature propaga-
tion network for video colorization. We develop a large-pretrained visual model
guided feature estimation module to better explore robust spatial features. To
establish reliable connections from far-apart frames, we propose a memory-based


```
ColorMNet: A MSTFP Network for Video Colorization 15
```
feature propagation module. We develop a local attention module to better uti-
lize spatial-temporal priors. Both quantitative and qualitative experimental re-
sults show that our method performs favorably against state-of-the-art methods.

## References

1. Chen, S., Li, X., Zhang, X., Wang, M., Zhang, Y., Han, J., Zhang, Y.: Exemplar-
    based video colorization with long-term spatiotemporal dependency. arXiv preprint
    arXiv:2303.15081 (2023)
2. Chen, X., Zou, D., Zhao, Q., Tan, P.: Manifold preserving edit propagation. ACM
    TOG 31 (6), 1–7 (2012)
3. Cheng, Z., Yang, Q., Sheng, B.: Deep colorization. In: ICCV (2015)
4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth
    16x16 words: Transformers for image recognition at scale. In: ICLR (2021)
5. Gao, R., Wang, L.: MeMOTR: Long-term memory-augmented transformer for
    multi-object tracking. In: ICCV (2023)
6. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
    In: CVPR (2016)
7. He, M., Chen, D., Liao, J., Sander, P.V., Yuan, L.: Deep exemplar-based coloriza-
    tion. ACM TOG 37 (4), 1–16 (2018)
8. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained
    by a two time-scale update rule converge to a local nash equilibrium. In: NeurIPS
    (2017)
9. Iizuka, S., Simo-Serra, E.: Deepremaster: temporal source-reference attention net-
    works for comprehensive video enhancement. ACM TOG 38 (6), 1–13 (2019)
10. Iizuka, S., Simo-Serra, E., Ishikawa, H.: Let there be color! joint end-to-end learning
of global and local image priors for automatic image colorization with simultaneous
classification. ACM TOG 35 (4), 1–11 (2016)
11. Kang, X., Lin, X., Zhang, K., et al.: Ntire 2023 video colorization challenge. In:
CVPRW (2023)
12. Kang, X., Yang, T., Ouyang, W., Ren, P., Li, L., Xie, X.: Ddcolor: Towards photo-
realistic image colorization via dual decoders. In: ICCV (2023)
13. Karen Simonyan, A.Z.: Very deep convolutional networks for large-scale image
recognition. In: ICLR (2015)
14. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: ICLR
(2015)
15. Lai, W.S., Huang, J.B., Wang, O., Shechtman, E., Yumer, E., Yang, M.H.: Learning
blind video temporal consistency. In: ECCV (2018)
16. Larsson, G., Maire, M., Shakhnarovich, G.: Learning representations for automatic
colorization. In: ECCV (2016)
17. Lei, C., Chen, Q.: Fully automatic video colorization with self-regularization and
diversity. In: CVPR (2019)
18. Lei, C., Xing, Y., Chen, Q.: Blind video temporal consistency via deep video prior.
In: NeurIPS (2020)
19. Levin, A., Lischinski, D., Weiss, Y.: Colorization using optimization. ACM TOG
23 (3), 689–694 (2004)
20. Liu, Y., Zhao, H., Chan, K.C., Wang, X., Loy, C.C., Qiao, Y., Dong, C.: Temporally
consistent video colorization with deep feature propagation and self-regularization
learning. arXiv preprint arXiv:2110.04562 (2021)


16 Y. Yanget al.

21. Luan, Q., Wen, F., Cohen-Or, D., Liang, L., Xu, Y., Shum, H.: Natural image
    colorization. In: ESRT (2007)
22. Oquab, M., Darcet, T., Moutakanni, T., et al.: Dinov2: Learning robust visual
    features without supervision. TMLR (2024)
23. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen,
    T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, high-
    performance deep learning library. In: NeurIPS (2019)
24. Paul, S., Bhattacharya, S., Gupta, S.: Spatiotemporal colorization of video using
    3d steerable pyramids. IEEE TCSVT 27 (8), 1605–1619 (2016)
25. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., Sorkine-
    Hornung, A.: A benchmark dataset and evaluation methodology for video object
    segmentation. In: CVPR (2016)
26. Qu, Y., Wong, T., Heng, P.: Manga colorization. ACM TOG 25 (3), 1214–
    (2006)
27. Sangkloy, P., Lu, J., Fang, C., Yu, F., Hays, J.: Scribbler: Controlling deep image
    synthesis with sketch and color. In: CVPR (2017)
28. Sheng, B., Sun, H., Magnor, M., Li, P.: Video colorization using parallel optimiza-
    tion in feature space. IEEE TCSVT 24 (3), 407–417 (2013)
29. Su, J.W., Chu, H.K., Huang, J.B.: Instance-aware image colorization. In: CVPR
    (2020)
30. Sun, G., Hua, Y., Hu, G., Robertson, N.: Mamba: Multi-level aggregation via
    memory bank for video object detection. In: AAAI (2021)
31. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In:
    ECCV (2020)
32. Thasarathan, H., Nazeri, K., Ebrahimi, M.: Automatic temporally coherent video
    colorization. In: CRV (2019)
33. Wan, Z., Zhang, B., Chen, D., Liao, J.: Bringing old films back to life. In: CVPR
    (2022)
34. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment:
    from error visibility to structural similarity. IEEE TIP 13 (4), 600–612 (2004)
35. Xu, Z., Wang, T., Fang, F., Sheng, Y., Zhang, G.: Stylization-based architecture
    for fast deep exemplar colorization. In: CVPR (2020)
36. Yang, Y., Peng, Z., Du, X., Tao, Z., Tang, J., Pan, J.: Bistnet: Semantic image
    prior guided bidirectional temporal feature fusion for deep exemplar-based video
    colorization. IEEE TPAMI pp. 1–14 (2024)
37. Yatziv, L., Sapiro, G.: Fast image and video colorization using chrominance blend-
    ing. IEEE TIP 15 (5), 1120–1129 (2006)
38. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer:
    Efficient transformer for high-resolution image restoration. In: CVPR (2022)
39. Zhang, B., He, M., Liao, J., Sander, P.V., Yuan, L., Bermak, A., Chen, D.: Deep
    exemplar-based video colorization. In: CVPR (2019)
40. Zhang, R., Isola, P., Efros, A.A.: Colorful image colorization. In: ECCV (2016)
41. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable
    effectiveness of deep features as a perceptual metric. In: CVPR (2018)
42. Zhang, R., Zhu, J.Y., Isola, P., Geng, X., Lin, A.S., Yu, T., Efros, A.A.: Real-time
    user-guided image colorization with learned deep priors. ACM TOG 36 (4), 1–
    (2017)
43. Zhao, H., Wu, W., Liu, Y., He, D.: Color2embed: Fast exemplar-based image col-
    orization using color embeddings. arXiv preprint arXiv:2106.08017 (2021)


```
ColorMNet: A MSTFP Network for Video Colorization 17
```
44. Zhao, Y., Po, L.M., Yu, W.Y., Rehman, Y.A.U., Liu, M., Zhang, Y., Ou, W.:
    Vcgan: video colorization with hybrid generative adversarial network. IEEE TMM
    (2022)


## ColorMNet: A Memory-based Deep

## Spatial-Temporal Feature Propagation Network

## for Video Colorization

```
Yixin Yang, Jiangxin Dong, Jinhui Tang, and Jinshan Pan
```
```
Nanjing University of Science and Technology
```
## Overview

```
In this document, we first present the network details in Section 1. Then, we
analyze the effectiveness of the proposed large-pretrained visual model guided
feature estimation (PVGFE) module and the memory-based feature propagation
(MFP) module in Section 2 and Section 3. To examine the effectiveness of the
proposed local attention (LA) module on video colorization, we further analyze
it in Section 4. In addition, we conduct a user study to investigate the subjective
preference by human observers of each colorization method in Section 5. Finally,
we show more visual comparisons on both synthetic datasets and real-world
videos in Section 6.
```
## 1 Network Details

```
As stated in Section 3 of the main manuscript, our method contains a large-
pretrained visual model guided feature estimation module, a memory-based fea-
ture propagation module, and a local attention module for video colorization.
We also show the network details of the proposed memory-based deep spatial-
temporal feature propagation network for video colorization in Figures 2 of the
main manuscript. In this document, we list the detailed architecture of our
proposed ColorMNet in Table 1. The spatial resolution of the input image is
448 × 448 pixels.
```
## 2 Effectiveness of the Large-Pretrained Visual Model

## Guided Feature Estimation Module

```
As stated in Section 5 of the manuscript, we have analyzed the effectiveness
of the large-pretrained visual model guided feature estimation (PVGFE) mod-
ule. In this supplemental material, we further show more visual comparisons
to demonstrate the effectiveness of the PVGFE module. In‘ComparisonsWith-
SOTA.mp4’, we show that our proposed ColorMNet with using the PVGFE is
able to generate better-colorized videos.
```
# arXiv:2404.06251v1 [cs.CV] 9 Apr 2024


2 Y. Yanget al.

Table 1:Detailed architecture of our proposed ColorMNet.

```
[
Conv. 7×7, 64, stride 2
```
```
]
```
denotes a convolution with the filter size of 7 × 7 pixels with the filter number of 64 with
stride 2, Embed dim. denotes the dimension of embedding,

```
[
Interpolation,× 2
```
]
denotes
an interpolation operation with a scale factor equal to 2,

```
[
ResBlock, 256
```
]
denotes a
ResBlock consisting of convolutions with the filter size of 3 × 3 pixels with the filter
number of 256.

```
ColorMNet
Output size ResNet50 [2] DINOv2 [8]
```
```
Key feature extractor (PVGFE)
```
```
28 × 28 × 1024
```
```
Conv. 7×7, 64, stride 2
MaxPool, 3 ×3, stride 2

```
```
Conv. 1×1, 64
Conv. 3×3, 64
Conv. 1×1, 256
```
```

× 3


```
```
Conv. 1×1, 128
Conv. 3×3, 128
Conv. 1×1, 512
```
```

× 4


```
```
Conv. 1×1, 256
Conv. 3×3, 256
Conv. 1×1, 1024
```
```

× 6
```
```
 Patch Embedding




```
```
Transformer block
Patch size = 14
Embed dim. = 384
Heads = 6
Blocks = 12
FFN layer = MLP
```
```





```
```
× 12
```
```
Concat features from last 4 layers
Conv. 1×1, 1536
Interpolation,× 14 / 16
Conv. 3×3, 1024
28 × 28 × 64 Conv. 3×3, 64 Conv. 3×3, 64
28 × 28 × 64 Cross-channel attention [11]
```
```
Value feature extractor (ResNet18 [2])^28 ×^28 ×^256
```
```
ResNet18 [2]
Conv. 7×7, 64, stride 2
MaxPool, 3[ ×3, stride 2
Conv. 1×1, 64
Conv. 3×3, 64
```
```
]
× 2
[Conv. 1×1, 128
Conv. 3×3, 128
```
```
]
× 2
[Conv. 1×1, 256
Conv. 3×3, 256
```
```
]
× 2
28 × 28 × 512 Conv. 3×3, 512
```
```
Decoder
```
```
112 × 112 × 256
```
```
Interpolation,× 2
ResBlock, 256
Interpolation,× 2
ResBlock, 256
112 × 112 × 2 Conv. 1×1, 2
112 × 112 × 2 GRU [1]
448 × 448 × 2 Interpolation,× 4
```
## 3 Effectiveness of the Memory-based Feature Propagation

## Module

As stated in Section 5 of the manuscript, we have analyzed the effectiveness
of the memory-based feature propagation (MFP) module. In this supplemental
material, we further show more visual comparisons to demonstrate the effective-
ness of the MFP module. In‘ComparisonsWithSOTA.mp4’, we show that our
proposed ColorMNet with using the MFP is able to generate better-colorized
videos compared with the method without using the MFP.

## 4 Effectiveness of the Local Attention Module

As stated in Section 5 of the manuscript, we have analyzed the effectiveness of the
local attention (LA) module. We empirically setλ= 7forλ×λpatchN(p). In


```
ColorMNet: A MSTFP Network for Video Colorization 3
```
```
ColorMNet (Ours)
58.5%
```
```
BiSTNet
25.5%
```
```
DeepExemplar
11.5%
```
```
DeepRemaster
4.5%
```
```
(b) Input video (c) Exemplar (d) [3]
```
```
(a) Averaged selection percentage of user study (e) [12] (f) [10] (g) Ours
```
Fig. 1:User study result and an example of a group of results displayed to human
observers in the user study. (a) shows that our proposed ColorMNet achieves obviously
higher score than other state-of-the-art methods, which demonstrates its subjective
advantages. (b)-(g) are the input video, the exemplar image, the colorized videos by
DeepRemaster [3], DeepExemplar [12], BiSTNet†[10] and ColorMNet (Ours), respec-
tively. We make the methods anonymous and randomly sort the videos in (d)-(g) to
ensure fairness.†denotes that two exemplars are used.

this supplemental material, we further show more visual comparisons to demon-
strate the effectiveness of the LA module. In‘ComparisonsWithSOTA.mp4’, we
show that our proposed ColorMNet with using the LA is able to generate better-
colorized videos.

## 5 User Study

To evaluate whether our results are favored by human observers, we further con-
duct user study experiments. Specifically, we compare our method with exemplar-
based methods, i.e., BiSTNet [10], DeepExemplar [12] and DeepRemaster [3]. We
randomly select 10 input videos from the DAVIS [9] validation set, the Videvo [6]
validation set and the NVCC2023 [4] validation set together with the coloriza-
tion results and the exemplar images displayed to 20 online observers without
constraints. We make the methods anonymous and randomly sort the videos
in each group to ensure fairness. Observers are asked to choose the most visu-
ally pleasing results from a group of videos. Figure 1 shows that our method is
preferred by a wider range of users than other state-of-the-art methods.

## 6 More Experimental Results

In this section, we provide more visual comparisons with state-of-the-art meth-
ods on both synthetic and real-world videos. Figures 2-13 show the compar-
isons, where our method generates better colorized frames. In‘Comparison-
sWithSOTA.mp4’, we show that the proposed method generates vivid and real-
istic videos.


4 Y. Yanget al.

## References

1. Cho, K., Van Merriënboer, B., Bahdanau, D., Bengio, Y.: On the proper-
    ties of neural machine translation: Encoder-decoder approaches. arXiv preprint
    arXiv:1409.1259 (2014)
2. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition.
    In: CVPR (2016)
3. Iizuka, S., Simo-Serra, E.: Deepremaster: temporal source-reference attention net-
    works for comprehensive video enhancement. ACM TOG 38 (6), 1–13 (2019)
4. Kang, X., Lin, X., Zhang, K., et al.: Ntire 2023 video colorization challenge. In:
    CVPRW (2023)
5. Kang, X., Yang, T., Ouyang, W., Ren, P., Li, L., Xie, X.: Ddcolor: Towards photo-
    realistic image colorization via dual decoders. In: ICCV (2023)
6. Lai, W.S., Huang, J.B., Wang, O., Shechtman, E., Yumer, E., Yang, M.H.: Learning
    blind video temporal consistency. In: ECCV (2018)
7. Liu, Y., Zhao, H., Chan, K.C., Wang, X., Loy, C.C., Qiao, Y., Dong, C.: Temporally
    consistent video colorization with deep feature propagation and self-regularization
    learning. arXiv preprint arXiv:2110.04562 (2021)
8. Oquab, M., Darcet, T., Moutakanni, T., et al.: Dinov2: Learning robust visual
    features without supervision. TMLR (2024)
9. Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., Sorkine-
    Hornung, A.: A benchmark dataset and evaluation methodology for video object
    segmentation. In: CVPR (2016)
10. Yang, Y., Peng, Z., Du, X., Tao, Z., Tang, J., Pan, J.: Bistnet: Semantic image
prior guided bidirectional temporal feature fusion for deep exemplar-based video
colorization. IEEE TPAMI pp. 1–14 (2024)
11. Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.: Restormer:
Efficient transformer for high-resolution image restoration. In: CVPR (2022)
12. Zhang, B., He, M., Liao, J., Sander, P.V., Yuan, L., Bermak, A., Chen, D.: Deep
exemplar-based video colorization. In: CVPR (2019)
13. Zhao, H., Wu, W., Liu, Y., He, D.: Color2embed: Fast exemplar-based image col-
orization using color embeddings. arXiv preprint arXiv:2106.08017 (2021)
14. Zhao, Y., Po, L.M., Yu, W.Y., Rehman, Y.A.U., Liu, M., Zhang, Y., Ou, W.:
Vcgan: video colorization with hybrid generative adversarial network. IEEE TMM
(2022)


```
ColorMNet: A MSTFP Network for Video Colorization 5
```
```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 2:Colorization results on clipbike-packingfrom the DAVIS validation dataset [9].
The results shown in (b) and (c) still contain significant color-bleeding artifacts. [3,
7, 12, 14] do not recover the man well. In contrast, our proposed method generates a
better-colorized frame, where the man is restored well and the colors look better.


6 Y. Yanget al.

```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 3:Colorization results on clipblackswanfrom the DAVIS validation dataset [9].
The results shown in (b) and (c) still contain significant color-bleeding artifacts. In
contrast, our proposed method generates a better-colorized frame.


```
ColorMNet: A MSTFP Network for Video Colorization 7
```
```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 4:Colorization results on clipbreakdancefrom the DAVIS validation dataset [9].
The results shown in (b) and (c) still contain significant color-bleeding artifacts. In
contrast, our proposed method generates a better-colorized frame, where the colors of
the dancer are restored well.


8 Y. Yanget al.

```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 5: Colorization results on clip car-roundabout from the DAVIS validation
dataset [9]. The results shown in (b) and (c) still contain significant color-bleeding
artifacts. In contrast, our proposed method restores the colors of the flowerbed and
generates a better-colorized frame.


```
ColorMNet: A MSTFP Network for Video Colorization 9
```
```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 6:Colorization results on cliploadingfrom the DAVIS validation dataset [9]. The
results shown in (b) and (c) still contain significant color-bleeding artifacts. In contrast,
our proposed method generates a vivid and realistic frame, where the colors of the box
and the man’s hands are better restored.


10 Y. Yanget al.

```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 7:Colorization results on clipCoupleRidingMotorbikefrom the Videvo validation
dataset [6]. The results shown in (b) and (c) still contain significant color-bleeding
artifacts. In contrast, our proposed method generates a realistic frame that is faithful
to the exemplar image.


```
ColorMNet: A MSTFP Network for Video Colorization 11
```
```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 8:Colorization results on clipCyclingfrom the Videvo validation dataset [6]. The
results shown in (b) and (c) still contain significant color-bleeding artifacts. In contrast,
our proposed method generates a vivid frame than other stage-of-the-art methods.


12 Y. Yanget al.

```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 9:Colorization results on clipSkateboarderTableJumpfrom the Videvo validation
dataset [6]. The results shown in (b) and (c) still contain significant color-bleeding
artifacts. In contrast, our proposed method generates a realistic frame, where the colors
of the skateboard man and the trees are better restored.


```
ColorMNet: A MSTFP Network for Video Colorization 13
```
```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 10:Colorization results on clipTimeSquareTrafficfrom the Videvo validation
dataset [6]. The results shown in (b) and (c) still contain significant color-bleeding
artifacts. In contrast, our proposed method generates a vivid and realistic frame against
other stage-of-the-art methods.


14 Y. Yanget al.

```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 11:Colorization results on clip 001 from the NVCC2023 validation dataset [4].
The results shown in (b) and (c) still contain significant color-bleeding artifacts. In
contrast, our proposed method generates a better-colorized frame, where the colors of
the woman and the leaves are better restored.


```
ColorMNet: A MSTFP Network for Video Colorization 15
```
```
(a) Input frame and exemplar image (b) DDColor [5]
```
```
(c) Color2Embed [13] (d) TCVC [7]
```
```
(e) VCGAN [14] (f) DeepRemaster [3]
```
```
(g) DeepExemplar [12] (h) ColorMNet (Ours)
```
Fig. 12:Colorization results on clip 014 from the NVCC2023 validation dataset [4].
The results shown in (b) and (c) still contain significant color-bleeding artifacts. In
contrast, our proposed method generates a vivid frame in (h) that is not only more
colorful compared with (d-g) but also faithful to the exemplar image in (a).


16 Y. Yanget al.

```
(a) Inputs (b) DeepRemaster [3] (c) DeepExemplar [12] (d) ColorMNet (Ours)
```
Fig. 13:Colorization results on real-world videos. From top to bottom are respec-
tively the film clip fromRoman Holiday (1953), the film clip fromMiracle on 34th
Street (1947), the film clip fromManhattan (1979)and a real-world video collected
from the internet. We obtain the exemplars by searching the internet to find the most
visually similar images to the input video frames. DeepRemaster [3] cannot gerenate
vivid frames. The results shown in (c) generated by DeepExemplar [12] still contain
significant color-bleeding artifacts (the wall of the building and the skiing man) and
cannot maintain faithfulness to the given exemplar images (over-saturated colors on
the both the face of the man and the face of the woman). In contrast, our proposed
method generates vivid and realistic frames.



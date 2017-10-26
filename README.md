<link href="ek_bootstrap_md.css" rel="stylesheet"></link>
# LifeScience AI
Reference MEMO of LifeScience AI by Eli Kaminuma

## Keywords
- convolutional neural networks (CNNs)
- clinically significant (CS) 
- apparent diffusion coefficients (ADCs) 

## 学習方法

- Supervised Learning 教師あり学習
  - LDA　線形判別分析
  - SVM　サポートベクターマシーン
  - Logistic Regression　ロジスティック回帰
  - RandomForest　ランダムフォレスト
- Unsupervised Learning 教師なし学習
  - kmeans 
  - PCA　主成分分析
  - 階層型クラスター分析
  - SOM　自己組織化マップ
- Semi-supervised Learning　半教師あり学習

- Ensembl Learning　 アンサンブル学習
- Reinforcement Learning 強化学習 
- Transfer Learning 転移学習 
  - A novel end-to-end classifier using domain transferred deep convolutional neural networks for biomedical images.
  - Transfer Learning and Sentence Level Features for Named Entity Recognition on Tweets
  - TRANSFER LEARNING FOR MUSIC CLASSIFICATION AND REGRESSION TASKS,  Urbansound8K dataset. 
  - TRANSFER LEARNING FOR SEQUENCE TAGGING WITH HIERARCHICAL RECURRENT NETWORKS
  
- Meta Learning メタ学習
  - [学習方法を学ぶ](https://medium.com/intuitionmachine/machines-that-search-for-deep-learning-architectures-c88ae0afb6c8) 
  - bias-variance dilemma/tradeoff = simple models (bias大,variance小),　complex models (bias小,variance大) 
  - meta reinforcement learning
  
- Deep learning for structured data creation データ構造化の深層学習 
   - [Snokel](https://github.com/HazyResearch/snorkel) 
- Attention RNNs 
  - [Domain Attention with an Ensemble of Experts](http://www.aclweb.org/anthology/P/P17/P17-1060.pdf)
  - [Neural Relation Extraction with Multi-lingual Attention](http://www.aclweb.org/anthology/P/P17/P17-1004.pdf)
- Multimodal Deep Learning
  - [Multimodal Deep Learning by Andrew Ng 2011](http://mfile.narotama.ac.id/files/Umum/JURNAR%20STANFORD/Multimodal%20deep%20learning.pdf)
   - zero-shot learning by [CVPR17 tutorial](http://isis-data.science.uva.nl/tmensink/docs/ZSL17.web.pdf)
   - generating visual explanations by Hendricks et.al. ECCV’16
  
| Ng's types  | Feature Learning  | Supervised Training   | Testing   |   
|---|---|---|---|
| Classic Deep Learning  | Audio   | Audio   | Audio  |   
| Classic Deep Learning  | Video   | Video   | Video  |  
| Multimodal Fusion  | A+V   |  A+V   |  A+V  |  
| Cross Modality Learning |  A+V  |  Video | Video   |   
| Cross Modality Learning |  A+V  |  Audio | Audio   |   
| Shared Representation Learning   | A+V   | Audio   | Video   |         
| Shared Representation Learning   | A+V   | Video   | Audio   |        
- Multimodal Data Fusion : [Survey2015](https://hal.archives-ouvertes.fr/hal-01179853/file/Lahat_Adali_Jutten_DataFusion_2015.pdf)
- Deep Probabilistic Programming [TOOLS]
  - TOOLS
    - [Edward](http://edwardlib.org/) (tensorflow backend, VI=BBVI, MCMC=MH/HMG/SGLD)
    - [PyMC3]() (theano backend, VI=ADVI, MCMC=MH/HMC/NUTS)
        - BBVI=Blackbox Variational Inference, ADVI=Automatic Differentiation Variational Inference
        - MH=Metropolis Hastings, HMC=Hamilton Monte Carlo, SGLD=Stochastic Gradient Langevin Dynamics
        - NUTS= No-U-Turn Sampler
    - [zhusuan](https://github.com/thu-ml/zhusuan?utm_content=buffer12448&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer)    
    - Stan, Anglican, Church, venture, Figaro, WebPPL
- Deep Probabilistic Programming [PAPERS]
     - https://arxiv.org/pdf/1701.03757.pdf
- Deep Probabilistic Programming [SAMPLES]
     - [混合ガウスモデル](http://s0sem0y.hatenablog.com/entry/2017/07/01/090657)
     - [Bayesian DNN＋Variational Inference](http://mathetake.hatenablog.com/entry/2017/01/19/134054)
     - Edward = Tensorflow + PPL
     - ベイジアンリカレントネット, ベイズ推定, MCMC, 変分推定 
     - http://edwardlib.org/tutorials/
     - ![Edward構造](https://cdn-ak.f.st-hatena.com/images/fotolife/x/xiangze/20170801/20170801071255.png "Edward")

## Taxonomy of Multimodal Machine Learning (MMML) 

- Representation
   - ___Joint___ (___NN___, Graphical models, Sequential)
   - Coordinated (Similarity, Structured)
- Translation
   - Example-based (Retrieval, Combination)
   - Model-based (Grammar-based, Encoder-decoder, Online prediction)
- Alignment
   - Explicit (Unsupervised, Supervised)
   - ___Implicit___ (Graphical models, ___NN___ )
- Fusion
   - Model agnostic (Early fusion, Late fusion, Hybrid fusion)
   - ___Model-based___ (Kernel-based, Graphical models, ___NN___)
- Co-learning 
    - Parallel data
       - Co-training
       - Transfer learning
    - Non-parallel data
       - zero-shot learning
       - concept grounding
       - transfer learning
    - Hybrid data
       - Bridging
 - Reference
     - [Multimodal Machine Learning:A Survey and Taxonomy from CMU Baltrusaitis et al.,2017](https://arxiv.org/abs/1705.09406)
     - https://arxiv.org/abs/1705.09406 Multimodal Applications
     - https://www.cs.cmu.edu/~morency/MMML-Tutorial-ACL2017.pdf

## MMMLアプリケーション

- [X-FIDO:An Effective Application for Detecting Olive Quick Decline Syndrome with Deep Learning and Data Fusion](https://www.frontiersin.org/articles/10.3389/fpls.2017.01741/full)

| PMID  | YEAR  | NOTE | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| 28582269   | 2017   |   |  | images | MM-CNN | 95.7% for indolent PCa,97.8% for CS lesions (AUC)  |   

## DNNフレームワーク

Python
- :us:Caffe, Caffe2 (UC Berkeley, Berkeley Vision and Learning Center ) 
- :us:Tensorflow (google)
- :canada:Theano (Université de Montréal)
- Keras -- Backend(TheanoかTensorflow)
- Edward -- Backend(Tensorflow) 
- :jp:Chainer (Preferred Networks)
- :us:CNTK (microsoft)
- :cn:Paddle (baidu)
- :uk:Sonnet (DeepMind)
- PyTorch

Lua
- Torch

JVM
- DL4J

## CNN Architectures

- ILSVRC'12 = [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (:canada: Univ of Tronto, Prof. Hinton) = 8 layers
- ILSVRC'14 = [VGG](http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/) (:uk:Univ of Oxford, VGG Group) = 19 layers
- ILSVRC'14 = GoogLeNet (:us:Google)      = 22 layers
- ILSVRC'15 = ResNet (:us:Microsoft Research Asia)  = 152 layers
- [アーキテクチャ精度解説資料](http://www.nlab.ci.i.u-tokyo.ac.jp/pdf/ieice201705cvcomp.pdf) = 画像解析関連コンペティションの潮流 中山英樹 信学会 100:373,2017

## DNN文献リスト(from PubMed, etc.)
- [Deep Patient](https://www.ncbi.nlm.nih.gov/pubmed/27185194)
    - electronic health records (EHRs)から、患者の将来のDisease Riskを予測(AUC=0.773)
    - using 76,214 test patients comprising 78 diseases 
- 健康質問応答サービス
- [28606870](https://www.ncbi.nlm.nih.gov/pubmed/28606870) 2017  オンライン健康質問応答サービスの品質予測, 90日間3人医師でラベル付け
  multimodal deep belief network (DBN) both textual features and non-textual features 
- 病理組織分類
  - 2017 横紋筋肉腫の病理組織学的サブタイプ分類, 転移学習, Multimodal(拡散強調MRスキャン（DWI）とガドリニウムキレート強化T1強調MRスキャン（MRI）の融合)
  - 2017 [腎臓のsegmentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5435691/)
- 微生物分類
  - Deep learning approach to bacterial colony classification. DIBaS dataset ( 660 images with 33 different genera and species of bacteria.)
- 植物葉分類
  - Automated classification of tropical shrub species: a hybrid of leaf shape and machine learning approach.
     - myDAUN dataset. 98.23%
     - the Flavia dataset  95.25%
     - Swedish Leaf dataset  99.89%
 -     
     - TITER: predicting translation initiation sites by deep learning.
     - DextMP: deep dive into text for predicting moonlighting proteins.
     - DeepCNF: AUC-Maximized Deep Convolutional Neural Fields for Protein Sequence Labeling

## Plant Cultivar Classification

| PMID  | YEAR  | Classification Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|
| 28857245  | 2017   |  16 European faba bean cultivars | 20 root traits  | k-NN  | 84.5% (Accuracy)  |  
| 27999587   | 2016   | 16 European Pisum sativum (pea) cultivars  | 36 root traits  | SVM,RF  | 86% of pairs (Accuracy)  |  
| 26669182  | 2015   | 4 pommelo cultivars   | leaf hyperspectral images  | BPNN,LS-SVM  | 97.92% (Accuracy)  |  
| 23857260  | 2013   | 4 rice cultivars   | seed hyperspectral images  | SVM,RF,PLSDA  | 80% (Accuracy)  |  
| 22957050  | 2012   | 10 olive cultivars   | RAPD,ISSR markers | SVM,NB,RF  | 70% (Accuracy) |   

## Plant Trait Prediction

| PMID  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
|  28585253    | 2017 | Bean | canned black bean texture | Hyper spectral images |PLSR | |
| 28574705  | 2017   | Apple  |Usage, Age, and Harvest Season | Biochemical Profile  | ---  | --- (Accuracy)  |  
| 28857245  | 2017   | Faba Bean  | North/South, KSC | Root Traits| RF, k-NN  | 84.5% (Accuracy)  |  
|  28386178  | 2017 | Crop | rice yield | GIS,soil,meteological factor |SVM | 85% (F1)|
| 28405214 |2016|Soybean |  Plant stress severity rating | RGB Image | classification trees| 96% (Accuracy)  |  


## Plant Disease Prediction

| PMID  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| 28757863   | 2017   | Apple  |  disease severity classification  | apple leaf black rot images in PlantVillage dataset  | CNN-VGG16 | 90.4% (Accuracy)  |  
| 28574705  | 2017   | Rice  | rice blast disease  | 6 weather variables   | BPNN | 65.42% (Accuracy)  |  

## Plant Seed Classification


| CITATION  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| J. Phys.: Conf. Ser. 803 012177 | 2017   | wheat, rapeseed, phacelia, flax, white mastard | 9 crop taxons  | images  | DNN | 95% (Accuracy)  | 
| PMID: 28420197 | 2017   | chinese cabbage | seed qualify  | images  | BPNN | 90.38% (Accuracy)  | 
|   | 2013   | [REVIEW](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.429.1008&rep=rep1&type=pdf) | seed classification   | images  | ---- | ---% (Accuracy)  | 
| PMID:  | 2015   | weed | seed   | 3980 images  | PCANet | 90.96% (Accuracy)  | 
| PMID:  | 2012   | chickpea | seed   | 400 images  | BPNN,SOM | 79% (Accuracy)  | 
| PMID:  | 2010   | cotton | seed   | 400 images  | BPNN,DT | 79% (Accuracy)  | 

## Plant Seed Storage Protein Content Prediction

| CITATION  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| PMID: 26139889| 2015   |  rice, wheat, maize, castor bean and thale cress| protein contents| seed AA protein sequences   | SVM | 91.3% (Accuracy)  | 


## Host Prediction

| PMID  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| 28361670  | 2017   | Bacteria  | 9 bacterial host genera  | 45 infectious viruses  | LR.SVM,RF | 85% (AUC)  |  

## Others

| PMID  | YEAR  | NOTE | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| arxiv.1704.03152 |2017 ｜CorrRNN (temporal model for temporal data) |  
| 26950929   | 2017   | EEG  | Patiant Cohort Discovery  | EEG signals + reports | MM-CNN | 70.43% (MAP)   |  
| 26950929   | 2017   | EEG  | Polarity Classification  | EEG signals + reports  | MM-CNN | 76.2% (F1)   |  
| arxiv.1703.08970 | 2017   | EEG+EMG  | 4 labels  | EEG signals + EMG signals  | MM-CNN | 78.1% (Accuracy)   |
| 26950929   | 2016   | sigle-cell image  | DMSO,Cluster-A,B,C  | 40783(DMSO), 1988 (cluster A), 9765 (cluster B), and 414(cluster C) images | CNN | 93.4% (Accuracy)  |  
| 26950929   | 2014   |  |   | images | deep Boltzmann machine (DBM)-based27 multimodal learning
model | 25.4% (MAP)  | 

## 大規模データセット
情報科学
- ImageNet
- Microsoft [COCO dataset](http://cocodataset.org/) - 330K images (>200K labeled), 1.5M object instances, 80 object categories
- ISI dataset
       - 11 subjects perform seven actions related to an insulin selfinjection activity. 
       - The dataset includes egocentric video data acquired using a Google Glass wearable camera, and motion data acquired using an Invensense motion wrist sensor.
- CMU-MOSI: Multimodal Opinion Sentiment Intensity 
        - video opinions from youtube movie reviews

ライフサイエンス
- [ANDI:The Alzheimer’s Disease Neuroimaging Initiative](http://adni.loni.usc.edu/) 
       - 818 ADNI participants (at the time, 128 with AD, 415 with MCI, 267 controls and 8 of uncertain diagnosis)
       - Illumina Omni 2.5M genome-wide association study (GWAS) single nucleotide polymorphism (SNP) arrays
-  [CADDementia](https://caddementia.grand-challenge.org/) 
      - Computer-Aided Diagnosis of Dementia based on structural MRI data.
      - data collected from three different sites, total 384  MRI scans 
-  [Temple University Hospital (TUH) EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/html/overview.shtml)
     -  ver13 :  over 25,000 sessions and 15,000 patients collected over 12 years 
     -  20MB of raw data, European Data Format (EDF+) file schema

## Webサービス
- [WebDNN](https://mil-tokyo.github.io/webdnn/ja/) (東大原田・牛久研究室)
-　[Google Teachable Machine](https://teachablemachine.withgoogle.com/) Google提供。PCカメラで深層学習のデモが出来る。

## Google(Tensorflow/etc) 
- [Deeplarningjs.org](https://deeplearnjs.org/demos/) web-based machine learning(WebGL)
- [Tensorboard API](https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins)
- [Tensorboard] plugin example](https://github.com/tensorflow/tensorboard-plugin-example)
- [MultiModel: Multi-Task Machine Learning Across Domains by Google Research Blog, June 2017](https://research.googleblog.com/2017/06/multimodel-multi-task-machine-learning.html)

## Link多数掲載のまとめサイト
 - [NVIDIAのDNNフレームワークリスト](https://developer.nvidia.com/deep-learning-frameworks)
 - [PocketCluster](https://blog.pocketcluster.io/page/6/) 
 - [Papers Deep Learning for Recommender System](http://shuaizhang.tech/2017/03/13/Papers-Deep-Learning-for-Recommender-System/)
 
## 深層学習用のNotePC
 - [DELL Alienware](http://www.dell.com/jp/business/p/laptops#!dlpgid=alienware-laptops?~ck=bt)
      - GeForce GTX 1060/1070搭載 NEW ALIENWARE 17
      - GeForce GTX 1060/1070搭載 NEW ALIENWARE 15     
 - [HP OMEN]
      - OMEN HP15-ce000 GeForce GTX 1060
      - OMEN HP17-an000 GeForce GTX 1060
      - OMEN HP17-an000 GeForce GTX 1070
      - OMEN X HP17 GeForce GTX 1080
 
## GPUカードの深層学習の性能比較

 - [Benchmark Report from HPC Systems](http://www.hpc.co.jp/benchmark20160610.html)
      - GPU card: NVIDIA® GeForce® GTX 1080
      - CNN Architectures: GoogLeNet and AlexNet 
 

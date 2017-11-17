<link href="ek_bootstrap_md.css" rel="stylesheet"></link>

## LifeScience AI by Eli Kaminuma
- Reference MEMO of LifeScience AI

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

## MMML Applications

- [X-FIDO:An Effective Application for Detecting Olive Quick Decline Syndrome with Deep Learning and Data Fusion](https://www.frontiersin.org/articles/10.3389/fpls.2017.01741/full)

| PMID  | YEAR  | NOTE | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| 28582269   | 2017   |   |  | images | MM-CNN | 95.7% for indolent PCa,97.8% for CS lesions (AUC)  |   

## DNN Frameworks

Python
- :us:Caffe, Caffe2 (UC Berkeley, Berkeley Vision and Learning Center ) 
- :us:Tensorflow (google)
- :canada:Theano (Université de Montréal)
- Keras -- Backend(TheanoかTensorflow)
- Edward -- Backend(Tensorflow) 
- :jp:Chainer (Preferred Networks)
- :us:CNTK (Microsoft)
- :cn:Paddle (Baidu)
- :uk:Sonnet (DeepMind)
- :us:PyTorch (Facebook)

Lua
- :us:Torch (Facebook)

JVM
- DL4J

## ML Model Architecture
### CNN Architectures

- ILSVRC'12 = [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (:canada: Univ of Tronto, Prof. Hinton) = 8 layers
- ILSVRC'14 = [VGG](http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/) (:uk:Univ of Oxford, VGG Group) = 19 layers
- ILSVRC'14 = GoogLeNet (:us:Google)      = 22 layers
- ILSVRC'15 = ResNet (:us:Microsoft Research Asia)  = 152 layers
- [アーキテクチャ精度解説資料](http://www.nlab.ci.i.u-tokyo.ac.jp/pdf/ieice201705cvcomp.pdf) = 画像解析関連コンペティションの潮流 中山英樹 信学会 100:373,2017

### Other NN Architectures
- [Recursive Cortical Network (RCN)](https://www.vicarious.com/2017/10/26/common-sense-cortex-and-captcha/)



## Autoencoder

- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)　解説
     - What are autoencoders?
     - Are they good at data compression?
     - What are autoencoders good for?
        -  data denoising 
        -  dimensionality reduction
     - So what's the big deal with autoencoders?
     - Let's build the simplest possible autoencoder
     - Adding a sparsity constraint on the encoded representations
     - ___Deep autoencoder___ = 多層
     - ___Convolutional autoencoder___
     - Application to ___image denoising___
     - ___Sequence-to-sequence autoencoder___
     - ___Variational autoencoder (VAE)___

## 画像変換手法
 - [Generative Adversarial Networks (GAN)](https://deephunt.in/the-gan-zoo-79597dc8c347) 
     - a branch of unsupervised learning
     - [Generative Models overview](https://blog.openai.com/generative-models/) by OpenAI 2016.6 
     - [GAN overview](https://www.kdnuggets.com/2017/01/generative-adversarial-networks-hot-topic-machine-learning.html) by KDNuggets 2017.1
     - Conditional GAN　- pix2pix (2016.11)
     - [Stack GAN](https://arxiv.org/abs/1612.03242) (2016.12) : 2階層(StageI,II)のGANモデル
     - [GANまとめ](https://qiita.com/eve_yk/items/f4b274da7042cba1ba76)
　
 - DCGAN(Deep Convolutional Generative Adversarial Networks) 
     - GANにCNNを適用
     - [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
     - [日本語解説](https://qiita.com/shu223/items/b6d8dc1fccb7c0f68b6b)
     
 - [CycleGAN](https://github.com/junyanz/CycleGAN) (UC barkley) : unpaired training data
     - demo images : from apple to orange (fruit box, fruits in trees)
     - demo images : from orange to apple
 - [pix2pix](https://phillipi.github.io/pix2pix/)
      - [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow)
      - [pix2pix-keras](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)
      - [chainer-pix2pix](https://github.com/mattya/chainer-pix2pix)

## 植物分野ツール
- [Deep Plant Phenomics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5500639/) Open Source Platform for Plant Phenotyping
      - leaf counting
      - mutant classification
      - age regression for top-down images of plant rosettes
    
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
- [Deep-Learning-Based Approach for Prediction of Algal Blooms](http://www.mdpi.com/2071-1050/8/10/1060)
- 植物葉分類
  - Automated classification of tropical shrub species: a hybrid of leaf shape and machine learning approach.
     - myDAUN dataset. 98.23%
     - the Flavia dataset  95.25%
     - Swedish Leaf dataset  99.89%
 -     
     - TITER: predicting translation initiation sites by deep learning.
     - DextMP: deep dive into text for predicting moonlighting proteins.
     - DeepCNF: AUC-Maximized Deep Convolutional Neural Fields for Protein Sequence Labeling

## Plant Species Prediction

- PlantCLEF
- [Deep Fruit Detection in Orchards](https://arxiv.org/pdf/1610.03677.pdf) 
     - 1120 Apples, 1964 mangoes, 620 Almonds
     - Object Recognition 
     - F1-score (Apple,Mango,Almond=0.904,0.908,0.775)
     - [Dataset Examples](http://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/)

## Plant Cultivar Classification

| CITATION  | YEAR  | Classification Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|
| PMID:28857245  | 2017   |  16 European faba bean cultivars | 20 root traits  | k-NN  | 84.5% (Accuracy)  |  
| PMID:27999587   | 2016   | 16 European Pisum sativum (pea) cultivars  | 36 root traits  | SVM,RF  | 86% of pairs (Accuracy)  |  
| PMID:26669182  | 2015   | 4 pommelo cultivars   | leaf hyperspectral images  | BPNN,LS-SVM  | 97.92% (Accuracy)  |  
| PMID:23857260  | 2013   | 4 rice cultivars   | seed hyperspectral images  | SVM,RF,PLSDA  | 80% (Accuracy)  |  
| PMID:22957050  | 2012   | 10 olive cultivars   | RAPD,ISSR markers | SVM,NB,RF  | 70% (Accuracy) |   

## Plant Trait Prediction

| CITATION  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| arXiv:1707.02290    | 2017 | Maize | tassels count | 361 feld images | CNN, modified VGG | 6.6(MAE) |
| PMID:28425947     | 2017 | Tomato | fruit count | train 24,000 synthetic images, test 100 real images | CNN,modified ResNet | 91%(Accuracy) |
| PMID:28585253    | 2017 | Bean | canned black bean texture | Hyper spectral images |PLSR | |
| PMID:28574705  | 2017   | Apple  |Usage, Age, and Harvest Season | Biochemical Profile  | ---  | --- (Accuracy)  |  
| PMID:28857245  | 2017   | Faba Bean  | North/South, KSC | Root Traits| RF, k-NN  | 84.5% (Accuracy)  |  
| PMID:28386178  | 2017 | Crop | rice yield | GIS,soil,meteological factor |SVM | 85% (F1)|
| PMID:28405214 |2016|Soybean |  Plant stress severity rating | RGB Image | classification trees| 96% (Accuracy)  |  


## Plant Disease Prediction

| CITATION  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| PMID:28757863   | 2017   | Apple  |  disease severity classification  | apple leaf black rot images in PlantVillage dataset  | CNN-VGG16 | 90.4% (Accuracy)  |  
| PMID:28574705  | 2017   | Rice  | rice blast disease  | 6 weather variables   | BPNN | 65.42% (Accuracy)  |  

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
| PMID:26139889| 2015   |  rice, wheat, maize, castor bean and thale cress| protein contents| seed AA protein sequences   | SVM | 91.3% (Accuracy)  | 

## Bacterial Classification

CITATION  | YEAR  | MEMO | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| PMID: 28910352  | 2017   | DIBaS dataset | species | bacterial colony images   | SVN | 97.2(Accuracy)  | 
| PMID: 27814364 | 2016   | Deep Learning Automates the Quantitative Analysis of Individual Cells in Live-Cell Imaging Experiments.|  multiple cell types  | microscope live cell image(fluorescence,phase,DIC)  |  Conv-net | 0.89(Jaccard Index) | 
| BCB15,978-1-4503-3853-0 | 2015   | | segmenting to bacterial colonies, agar, plate, artefacts | bacterial colony images   |  Convolutional Deep Belief Network (CDBN) |  unsupervised leraning| 


## Host Prediction

| CITATION  | YEAR  | TAXON | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| PMID:28361670  | 2017   | Bacteria  | 9 bacterial host genera  | 45 infectious viruses  | LR.SVM,RF | 85% (AUC)  |  

## Others

| CITATION  | YEAR  | NOTE | Prediction Target   | Input Data  | Models |  Performance |
|---|---|---|---|---|---|---|
| PMID： 29086034  | 2017.10| Kvasir dataset | |Endoscopic images | CNN-AlexNet [:link:](https://github.com/jso111/ELHnet/tree/master/elhnet)  |  | 
| PMID： 29083930 | 2017.10|  ||Endoscopic images | VGG, AlexNet| | 
| PMID： 29083930 | 2017.10| | | OCT images | VGG16,scratch| | 
| PMID: 29065651 | 2017.9| LIDC-IDRI dataset | Lung Cancer | CT images | VGG16,scratch| 84.15(Accuracy)| 
| arxiv.1612.02572 |2016 ｜ | | brain age | brain MRI image | CNN-GM | 4.16(MAE years)|   
| arxiv.1704.03152 |2017 ｜||  | |CorrRNN (temporal model for temporal data) | |  
| PMID:26950929   | 2017   | EEG  | Patiant Cohort Discovery  | EEG signals + reports | MM-CNN | 70.43% (MAP)   |  
| PMID:26950929   | 2017   | EEG  | Polarity Classification  | EEG signals + reports  | MM-CNN | 76.2% (F1)   |  
| arxiv.1703.08970 | 2017   | EEG+EMG  | 4 labels  | EEG signals + EMG signals  | MM-CNN | 78.1% (Accuracy)   |
| PMID:26950929   | 2016   | sigle-cell image  | DMSO,Cluster-A,B,C  | 40783(DMSO), 1988 (cluster A), 9765 (cluster B), and 414(cluster C) images | CNN | 93.4% (Accuracy)  |  

### REVIEW PAPER
- [Machine Learning for High-Throughput Stress Phenotyping in Plants](http://www.sciencedirect.com/science/article/pii/S1360138515002630)

## 大規模データセット
情報科学
- [ImageNet](https://www.image-net.org/) (:us:Princeton Univ)
- Microsoft [COCO dataset](http://cocodataset.org/) - 330K images (>200K labeled), 1.5M object instances, 80 object categories
- ISI dataset
       - 11 subjects perform seven actions related to an insulin selfinjection activity. 
       - The dataset includes egocentric video data acquired using a Google Glass wearable camera, and motion data acquired using an Invensense motion wrist sensor.
- CMU-MOSI: Multimodal Opinion Sentiment Intensity 
        - video opinions from youtube movie reviews
- [CMP facade dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/)        

ライフサイエンス
- [Kvasir dataset](http://datasets.simula.no/kvasir/) 
      - Endoscopic images
      - Kvasir dataset v1 : 4,000 images, 8 classes (500 per class), JPEG
      - Kvasir dataset v2 : 8,000 images, 8 classes, (1,000 per class), JPEG
      - collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway
      - The VV consists of 4 hospitals and provides health care to 470.000 people. 
      - annotated and verified by medical doctors (experienced endoscopists)
      -  images with different resolution from 720x576 up to 1920x1072 pixels
- [LIDC-IDRI dataset](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#57a090df52b042cc99990ded8c89cbb8)   
      - Lung Cancer 
      - 1018 CT, 290 CR/DX
      - 244,527 images
      - Lung Image Database Consortium
      - marked lesions belonging to one of three categories ("nodule > or =3 mm," "nodule <3 mm," and "non-nodule > or =3 mm"). 
- [ANDI:The Alzheimer’s Disease Neuroimaging Initiative](http://adni.loni.usc.edu/) 
      - 818 ADNI participants (at the time, 128 with AD, 415 with MCI, 267 controls and 8 of uncertain diagnosis)
      - Illumina Omni 2.5M genome-wide association study (GWAS) single nucleotide polymorphism (SNP) arrays
-  [CADDementia](https://caddementia.grand-challenge.org/) 
      - Computer-Aided Diagnosis of Dementia based on structural MRI data.
      - data collected from three different sites, total 384  MRI scans 
-  [Temple University Hospital (TUH) EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/html/overview.shtml)
     -  ver13 :  over 25,000 sessions and 15,000 patients collected over 12 years 
     -  20MB of raw data, European Data Format (EDF+) file schema
- [DiBaS Dataset](http://misztal.edu.pl/software/databases/dibas/)
     -  Bacterial colony classification.
     -  660 images with 33 different genera and species of bacteria.
- [IPPN Plant Phenotyping dataset](https://www.plant-phenotyping.org/datasets-home)
- [ACID Wheat spikes and spikelets dataset](https://github.com/mikepound/wheat-hg)

     
## Major Competitions
情報科学
 - [ILSVRC](https://www.image-net.org/challenges/LSVRC/): ImageNet Large Scale Visual Recognition Competition
 - ImageCLEF
      - ImageCLEFcaption
      - ImageCLEFlifelog
      - ImageCLEFtuberculosis
      - visualQuestionAnswering
      - Medical Task
      - Image Annotation Task
      - Handwritten Retrieval 
      - Robot vision
      - Liver CT annotation
      - Domain adaptation
      - Plant identification
 - LifeCLEF
      - PlantCLEF
      - SeaCLEF
      - BirdCLEF
      - GeoLifeCLEF
      - ExperLifeCLEF

### Conference/Workshop
- [COMPUTER VISION PROBLEMS IN PLANT PHENOTYPING (CVPPP)](https://www.plant-phenotyping.org/CVPPP2017)
 
## Webサービス for Deep Learning
- [WebDNN](https://mil-tokyo.github.io/webdnn/ja/) (東大原田・牛久研究室)
-　[Google Teachable Machine](https://teachablemachine.withgoogle.com/) Google提供。PCカメラで深層学習のデモが出来る。
- [Sony Neural Network Console - Cloud](https://dl.sony.com/ja/cloud/)
   　- 無料はファイル容量 10GB、使用可能時間 10H、プロジェクト数 10Pまで。

## Smartphone Service for Lifescience AI
- [PlantNet Plant Identification](https://identify.plantnet-project.org/)
- [DeepFish](https://getdeepfish.com/#/)
- [Plantix](https://plantix.net/ja)
     - AI driven disease detection
- [Garden Answers Plant Id](http://www.gardenanswers.com/)     

## Google(Tensorflow/etc) 
- [Deeplarningjs.org](https://deeplearnjs.org/demos/) web-based machine learning(WebGL)
- [Tensorboard API](https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins)
- [Tensorboard] plugin example](https://github.com/tensorflow/tensorboard-plugin-example)
- [MultiModel: Multi-Task Machine Learning Across Domains by Google Research Blog, June 2017](https://research.googleblog.com/2017/06/multimodel-multi-task-machine-learning.html)

## Link多数掲載のまとめサイト
 - [Biological datasets for machine learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research#Biological_data) 
 - [NVIDIAのDNNフレームワークリスト](https://developer.nvidia.com/deep-learning-frameworks)
 - [PocketCluster](https://blog.pocketcluster.io/page/6/) 
 - [Papers Deep Learning for Recommender System](http://shuaizhang.tech/2017/03/13/Papers-Deep-Learning-for-Recommender-System/)
 - ["Deep Learning" AND Plant ](https://www.readbyqxmd.com/keyword/162049)
 - [deeplearning-biology](https://github.com/hussius/deeplearning-biology)
 - [List of deep learning implementations in biology](https://followthedata.wordpress.com/2015/12/21/list-of-deep-learning-implementations-in-biology/)
  
## 深層学習用のNotePC
 - [DELL Alienware](http://www.dell.com/jp/business/p/laptops#!dlpgid=alienware-laptops?~ck=bt)
      - GeForce GTX 1060/1070搭載 NEW ALIENWARE 17
      - GeForce GTX 1060/1070搭載 NEW ALIENWARE 15     
 - [HP OMEN]
      - OMEN HP15-ce000 GeForce GTX 1060
      - OMEN HP17-an000 GeForce GTX 1060
      - OMEN HP17-an000 GeForce GTX 1070
      - OMEN X HP17 GeForce GTX 1080

## AI Chip
- NVIDIA 
- Intel : Loihi : 128 cores, 1024 neurons
- Apple :  Neural Engine : iphoneX  
- IBM : TrueNorth : 4096 cores, 100万neurons, 2億5600万個synapses

## GPUカードの深層学習の性能比較

 - [Benchmark Report from HPC Systems](http://www.hpc.co.jp/benchmark20160610.html)
      - GPU card: NVIDIA® GeForce® GTX 1080
      - CNN Architectures: GoogLeNet and AlexNet 
 
## Deep Learning Cloud Platform
  - [FloydHub](https://www.floydhub.com/) : Deep Learningを対象としたPlatform-as-a-Service(PAAS) Cloudサービス.
  
## AI Patent
　
  - [数の米国、攻める中国　AI特許6万件を解剖](https://vdata.nikkei.com/newsgraphics/ai-patent/) by 日経新聞2017.2.1
       - 世界のAI特許、2010年と比べ７割増
       - 米国1.26倍, 中国2.9倍, 日本3%減
       -　出願数トップ3 
           - 米国=IBM:3049,Microsoft:1866,Google:979
           - 中国=中国国家電網公司:757,北京大学:442,南京大学:385
           - 日本=NTT:567,NEC:541,日立:420
  - [AIビジネスにおける特許戦略](https://business.bengo4.com/category5/article153)
       - 学習用プログラムの法的保護
       - 学習済みモデルと法的保護
       - 蒸留モデル
       - 誰が著作者、発明者にあたるか
  - [AI特許の権利化での注意点](http://knpt.com/contents/ai/ai_consulting.html)
  - [ハイテク食品メーカーHampton CreekのAI特許問題](http://gigazine.net/news/20171002-stupid-patent/)
       - 電子フロンティア財団が毎月発表している「[Stupid Patent of the Month](https://www.eff.org/ja/taxonomy/term/11344)」
       - 食物に含まれる含有物を発見する特許
  - [第四次産業革命の中で知財システムに何が起きているか](http://www.meti.go.jp/committee/kenkyukai/sansei/daiyoji_sangyo_chizai/pdf/001_02_00.pdf) by 経済産業省2016.10
  - AI成果物の著作権
      - 学習用プログラムのコード:著作権法2条10号の2「プログラムの著作物」
      - アルゴリズム:特許法2条3項1号の「物の発明」
            
       
  

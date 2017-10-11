# LifeScience AI
Reference MEMO of LifeScience AI by Eli Kaminuma

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
  
|   | Feature Learning  | Supervised Training   | Testing   |   
|---|---|---|---|
| Classif Deep Learning  | Audio   | Audio   | Audio  |   
| Classif Deep Learning  | Video   | Video   | Video  |  
| Multimodal Fusion  | A+V   |  A+V   |  A+V  |  
| Cross Modality Learning |  A+V  |  Video | Audio   |   
| Cross Modality Learning |  A+V  |  Audio | Video   |   
| Shared Representation Learning   | A+V   | Audio   | Video   |         
| Shared Representation Learning   | A+V   | Video   | Audio   |         

## DNNフレームワーク

Python
- :us:Caffe, Caffe2 (UC Berkeley, Berkeley Vision and Learning Center ) 
- :us:Tensorflow (google)
- Theano
- Keras -- TheanoかTensorflowのラッパーライブラリ
- :jp:Chainer (Preferred Networks)
- :us:CNTK (microsoft)
- :cn:Paddle (baidu)
- MeNet
- PyTorch

Lua
- Torch

JVM
- DL4J

## DNN文献リスト(from PubMed, etc.)
- 健康質問応答サービス
- [28606870](https://www.ncbi.nlm.nih.gov/pubmed/28606870) 2017  オンライン健康質問応答サービスの品質予測, 90日間3人医師でラベル付け
  multimodal deep belief network (DBN) both textual features and non-textual features 
- 病理組織分類
 - 2017 横紋筋肉腫の病理組織学的サブタイプ分類, 転移学習, Multimodal(拡散強調MRスキャン（DWI）とガドリニウムキレート強化T1強調MRスキャン（MRI）の融合)
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
| 26669182  | 2015   | 4 pommelo cultivars   | leaf hyperspectral images  | PCA-LS-SVM  | 99.46% (Accuracy)  |  
| 23857260  | 2013   | 4 rice cultivars   | seed hyperspectral images  | SVM,RF,PLSDA  | 80% (Accuracy)  |  
| 22957050  | 2012   | 10 olive cultivars   | RAPD,ISSR markers | SVM,NB,RF  | 70% (Accuracy) |   


     
 ## Link多数掲載のまとめサイト
 - [NVIDIAのDNNフレームワークリスト](https://developer.nvidia.com/deep-learning-frameworks)
 - [PocketCluster](https://blog.pocketcluster.io/page/6/) 
 - [Papers Deep Learning for Recommender System](http://shuaizhang.tech/2017/03/13/Papers-Deep-Learning-for-Recommender-System/)
     

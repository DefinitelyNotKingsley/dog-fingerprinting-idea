# Dog Vocal Fingerprinting

This repository explores dog vocal fingerprinting using audio embeddings and metric-learning based identity matching.

## Goal

The goal is to build a dog fingerprinting system that can:

- learn dog identity embeddings from bark clips
- enroll dogs using a small number of bark samples
- identify known dogs using cosine similarity
- reject unseen dogs as unknown
- eventually integrate with visual tracking for multi-dog videos

## Current Pipeline

```text
DogSpeak bark clips (.wav)
↓
Feature Extraction
Dog2Vec + MFCC + ECAPA-TDNN
↓
Feature Fusion
Fusion MLP
↓
Metric Learning
ArcFace
↓
256-D Dog Identity Embedding
↓
Enrollment Profiles
average 5 barks per dog
↓
Cosine Similarity
↓
Known Dog / Unknown Dog
Features Used
Frozen Feature Extractors
Model	Dimension	Purpose
Dog2Vec	768	Dog vocal representation
MFCC	80	Handcrafted acoustic features
ECAPA-TDNN	192	Speaker verification features
Trainable Components
Model	Purpose
Fusion MLP	Combines all feature extractors
ArcFace	Learns a discriminative dog identity embedding space
Dataset

Experiments use the DogSpeak dataset.

Full dataset: 77,202 bark clips
Original dog identities: 156
After merging dog_7a and dog_7b: 155 identities
Repeated open-set experiments:
145 known dogs
10 held-out unknown dogs
5 repeated random holdouts
Main Experiments
Closed-Set Fingerprinting

The model is trained on all known dog identities and evaluated using enrollment profiles and cosine similarity.

True Open-Set Fingerprinting

For each repeated holdout:

Randomly hold out 10 dog identities.
Train Fusion MLP + ArcFace only on the remaining 145 dogs.
Enroll known dogs using 5 bark samples.
Test on both known dogs and held-out unknown dogs.
Reject a bark as unknown if cosine similarity is below a threshold.
Repeated 10-Dog Holdout Results

Five repeated 10-dog holdout experiments using 5-bark enrollment.

Threshold	Overall Accuracy (%)	Known Dog Accuracy (%)	Unknown Recall (%)	False Unknown Rate (%)
0.35	67.57 ± 5.67	72.38 ± 1.03	5.61 ± 1.24	1.94 ± 0.39
0.40	67.80 ± 5.04	71.68 ± 1.10	19.54 ± 3.42	7.27 ± 1.05
0.45	67.66 ± 4.29	70.18 ± 1.33	39.39 ± 6.91	14.65 ± 1.40
Interpretation

The system is stronger at identifying enrolled dogs than rejecting unseen dogs. Increasing the rejection threshold improves unknown dog detection, but also increases false unknown rejections.

A threshold around 0.40 gives the best overall trade-off in the current experiment.

Repository Structure
scripts/
  01_inspect_dataset.py
  02_check_full_embeddings.py
  03_prepare_data.py
  04_train_arcface_fusion.py
  05_eval_fingerprint.py
  06_extract_full_ecapa.py
  07_prepare_data_with_ecapa.py
  08_train_arcface_fusion3.py
  09_eval_fingerprint_fusion3.py
  10_train_attention_fusion_arcface.py
  11_eval_attention_fingerprint.py
  12_unknown_rejection_fusion3.py
  13_open_set_unknown_eval.py
  14_create_10dog_holdout_split.py
  15_train_holdout10_arcface.py
  16_embed_holdout10_unknown.py
  17_eval_holdout10_open_set.py
  18_repeated_holdout_validation.py

data/
  Prepared local arrays, not tracked by Git

models/
  Trained model checkpoints, not tracked by Git

results/
  Experiment outputs, not tracked by Git
Future Work
Fine-tune ECAPA-TDNN on DogSpeak instead of using frozen human speech embeddings.
Explore stronger metric-learning losses such as Supervised Contrastive Loss, Prototypical Networks, and Center Loss.
Build breed-balanced open-set benchmarks.
Improve unknown dog rejection.
Integrate audio fingerprinting with visual dog detection/tracking such as RT-DETR.
Long-Term Goal

Develop an open-set audio-visual dog fingerprinting system that can identify which visible dog barked in real-world multi-dog videos.

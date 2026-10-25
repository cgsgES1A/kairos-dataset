# Kairos Lab Report

### Author: Sopina Elizaveta

Classifying a post's performance relative to its author's usual level: `below`, `typical`, `above`. The primary metric is macro-F1.

## Methods Used in the Final Model

- **Tabular features:** post length, type and formatting, links, hashtags, mentions, emojis, letter case, punctuation; time and its cyclical encoding, post age, author profile and followers.
- **Author history:** reaction statistics, windows of 3/5/10/20 posts, trends, exponential averaging, deviations from usual formatting and posting rhythm. Additional history features use only strictly earlier posts.
- **Text:** normalization of anonymized identifiers; TF-IDF of words/bigrams and character 3-5-grams; text from posts and attachments.
- **Embeddings:** frozen multilingual MiniLM (384 features); splitting long texts into chunks, averaging and L2 normalization.
- **History retrieval:** BM25, cosine similarity of embeddings, statistics of similar past posts and alignment with the author's profile.
- **Images:** RGB histograms, frozen CLIP ViT-B/32 (512 features), averaging representations of images/previews from core.
- **Preprocessing:** median imputation, StandardScaler, One-Hot Encoding and LightGBM categorical features.
- **Base models:** LinearSVC; HistGradientBoostingRegressor for the deviation of `log(1 + reactions)` from the author's level; a LightGBM classifier; cosine kNN (`k=15`, class-wise sums of similarities).
- **Stacking:** logistic regression on OOF predictions from **3 author-based folds**, averaging base predictions; class weights and adjustments to log probabilities.
- **Reproducibility:** author-based splitting, fixed seeds and library/model versions, embedding caching.

## Rejected and Replaced Variants

Numerical hyperparameter searches are grouped. Checks included pilot experiments and full CV; not all excluded methods were initially unsuccessful.

| Method / Approach | What Was Tested and Why It Was Not Retained |
|---|---|
| SVD / LSA | TF-IDF compression; replaced by SVM, and subsequent checks did not bring the method back into the ensemble. |
| RBF / Nyström | Nonlinear transformation; did not improve the metrics. |
| Lemmatization, Unicode/HTML normalization | Additional text processing without a CV gain. |
| χ² selection, quantile discretization | Term selection and interval encoding of numerical values; not selected. |
| LDA | Topic probabilities: a small CV gain, but worse `evaluate` results. |
| VADER, pronouns, NLI emotion/intent features | Lexicon-based and semantic features without a sufficient gain. |
| Additional text context | Lexical similarity to history was reverted after `evaluate`; joint encoding and subtracting the author's mean vector did not justify the complexity. |
| SIFT / Bag of Visual Words | Averaged descriptors worsened CV; a visual vocabulary initially helped but was later removed during simplification because it did not provide a substantial gain. |
| Weighted late fusion | Text-based LR combined with boosting was replaced by stacking; a separate Random Forest on RGB/SIFT did not justify the added complexity. |
| Alternative classifiers | Random Forest, Extra Trees, CatBoost, Naive Bayes, NB-SVM, LR/MLP on embeddings, one-vs-rest and ordinal variants: not selected. |
| Other kNN variants | Neighbours based on TF-IDF/numerical features and replacing SVM were inferior to the selected approach of adding embedding-based kNN. |
| Specialists for `below` | Binary classification, reweighting hard examples, predicting 0/1/2+ reactions: no consistent macro-F1 gain. |
| SetFit / hard negatives | Contrastive fine-tuning, full text, pairs based on OOF errors, borderline examples: the gain did not justify the complexity. |
| Other encoders, OCR, supervised fine-tuning | English MiniLM, E5, MPNet, recognizing text in images and fine-tuning: the effect was small, absent or unconfirmed. |
| Network features | Link structure and neighbours based on organizations: worse performance or minimal improvement. |
| Visual relationships and a conditional branch | Text/image similarity, visual history/novelty, a separate branch for posts with images: the complexity was not justified. |
| RankNet | Ranking within each author: a CV gain of about 0.00024 was too small. |
| Ensemble simplifications | Duplicate context in the meta-model was removed; removing images or history retrieval was rejected because performance deteriorated. |
| 5 inner folds | `evaluate`: macro-F1 0.6618 → 0.6610; restored 3 folds. |

One issue affecting the metrics is authors with low popularity, for whom the reaction counts corresponding to different classes differ only slightly \
(for example, when an author's typical reaction count is 2). To address this issue, I tried various approaches (listed above), \
but none produced a substantial improvement. SetFit is the most promising (as well as, potentially, other algorithms for constructing hard negatives). \
The issue remains unresolved and requires further experimentation.

## Final Metrics

The standard `evaluate.py`, the same test set: **17 623 posts, 824 authors**; train: 71 395 posts, 3 295 authors.

| Metric | Baseline | Final Model |
|---|---:|---:|
| **macro-F1** | **0.5410** | **0.6618** |
| QWK | 0.4576 | 0.6037 |
| Accuracy | 0.5465 | 0.6991 |
| F1 `below` | 0.4859 | 0.5507 |
| F1 `typical` | 0.5689 | 0.7544 |
| F1 `above` | 0.5683 | 0.6802 |

The macro-F1 improvement over the baseline is **0.1208**.
Mean macro-F1 across **5 outer author-based CV folds - 0.6709**

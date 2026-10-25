'''
Upd 11.10: Max Verstappen wins Singapore Grand Prix!!!
TU TU DU DU
MAX VERSTAPPEN

Upd 25.10: Austin Grand Prix!
'''

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from PIL import Image, UnidentifiedImageError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from kairos import features
from kairos.data import DATA_DIR, load_followers, load_labels, load_posts
from kairos.label import author_level
from kairos.metrics import score
from my_model_features import (augment, HISTORY_COLUMNS, CALENDAR_COLUMNS, PROFILE_NUM,
                               PROFILE_CAT, RETRIEVAL_COLUMNS, CONTENT_EXTRA_NUM)
from my_model_ensemble import AuthorStackingClassifier, SortedCSR

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
STYLE_NUM = [
    "n_emoji_like", "n_emoticons", "n_exclamations", "n_repeated_punct",
    "uppercase_ratio", "n_elongated_words",
]
AUTHOR_CONTEXT_NUM = [
    "words_vs_author", "lines_vs_author", "type_history_share", "gap_vs_author",
]
IMAGE_NUM = ["image_available"] + [
    f"image_{channel}_{bin_index}" for channel in "rgb" for bin_index in range(8)
]
NUMERIC = (
    features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
    + STYLE_NUM + AUTHOR_CONTEXT_NUM + IMAGE_NUM
)
INPUTS = NUMERIC + features.CONTENT_CAT + ["text", "post_images"]

_EMOJI_BASE = (
    r"[\U0001F300-\U0001F3FA\U0001F400-\U0001F64F"
    r"\U0001F680-\U0001F6FF\U0001F7E0-\U0001F7EB"
    r"\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\u2600-\u27BF]"
)
_EMOJI_PART = _EMOJI_BASE + r"[\uFE0E\uFE0F]?[\U0001F3FB-\U0001F3FF]?"
_EMOJI = re.compile(
    r"(?:[\U0001F1E6-\U0001F1FF]{2}|[0-9#*]\uFE0F?\u20E3|"
    + _EMOJI_PART + r"(?:\u200D" + _EMOJI_PART + r")*)"
)
_EMOTICON = re.compile(
    r"(?:[:;=][-^']?[)(/\\]|[)(][-^']?[:;=]|<3|"
    r"[:;=][-^']?[DPpOo](?!\w)|(?<!\w)[8xX][-^']?[)(DPpOo](?!\w))"
)
_ELONGATED = re.compile(r"\b\w*([^\W\d_])\1{2,}\w*\b", re.IGNORECASE)


def add_style_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    text = table["text"].fillna("").str.replace(r"⟨[^⟩]*⟩", "", regex=True)
    table["n_emoji_like"] = text.str.count(_EMOJI)
    table["n_emoticons"] = text.str.count(_EMOTICON)
    table["n_exclamations"] = text.str.count("!")
    table["n_repeated_punct"] = text.str.count(r"[!?]{2,}")
    letters = text.map(lambda s: sum(c.isalpha() for c in s))
    uppercase = text.map(lambda s: sum(c.isupper() and c.isalpha() for c in s))
    table["uppercase_ratio"] = uppercase / letters.clip(lower=1)
    table["n_elongated_words"] = text.str.count(_ELONGATED)
    return table


def add_author_context(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out[AUTHOR_CONTEXT_NUM] = np.nan
    history = out.loc[out["posted_at"].notna()].sort_values(
        ["author", "posted_at"], kind="mergesort",
    )
    if history.empty:
        return out
    by_author = history.groupby("author", sort=False)
    first = ~history.duplicated(["author", "posted_at"])
    starts = np.flatnonzero(first.to_numpy())
    batch = first.to_numpy().cumsum() - 1

    for column, name in [("n_words", "words_vs_author"), ("n_lines", "lines_vs_author")]:
        past_median = by_author[column].transform(
            lambda s: s.shift(1).expanding(min_periods=1).median()
        )
        reference = past_median.to_numpy()[starts][batch]
        out.loc[history.index, name] = np.log1p(history[column].to_numpy()) - np.log1p(reference)

    earlier_count = by_author.cumcount().to_numpy()[starts][batch]
    same_type_count = (
        history.groupby(["author", "content_type"], sort=False, dropna=False).cumcount()
        - history.groupby(
            ["author", "posted_at", "content_type"], sort=False, dropna=False,
        ).cumcount()
    )
    out.loc[history.index, "type_history_share"] = (
        same_type_count.to_numpy() / np.where(earlier_count > 0, earlier_count, np.nan)
    )

    times = history.loc[first, ["author", "posted_at"]]
    gaps = times.groupby("author", sort=False)["posted_at"].diff().dt.total_seconds() / 86400
    past_gap = gaps.groupby(times["author"], sort=False).transform(
        lambda s: s.shift(1).expanding(min_periods=1).median()
    )
    gap_relative = np.log1p(gaps) - np.log1p(past_gap)
    out.loc[history.index, "gap_vs_author"] = gap_relative.to_numpy()[batch]
    return out


def add_image_features(table: pd.DataFrame) -> pd.DataFrame:
    files = sorted({file for group in table["post_images"] for file in group})
    histograms = {}
    failed = 0
    for file in tqdm(files, desc="image histograms", unit="file"):
        try:
            with Image.open(file) as image:
                image.thumbnail((128, 128), Image.Resampling.BILINEAR)
                rgba = image.convert("RGBA")
                rgb = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba,
                ).convert("RGB")
                pixels = np.asarray(rgb).reshape(-1, 3)
                histograms[file] = np.concatenate([
                    np.bincount(pixels[:, channel] // 32, minlength=8)
                    / len(pixels)
                    for channel in range(3)
                ])
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            failed += 1
    if failed:
        warnings.warn(
            f"Could not read {failed} of {len(files)} attached image files; "
            "using the remaining images or an unavailable-image indicator.",
            RuntimeWarning,
        )

    values = np.zeros((len(table), len(IMAGE_NUM)), dtype=np.float64)
    for row, files in enumerate(table["post_images"]):
        vectors = [
            histograms[file] for file in files
            if file in histograms
        ]
        if vectors:
            values[row, 0] = 1
            values[row, 1:] = np.mean(vectors, axis=0)
    out = table.copy()
    out[IMAGE_NUM] = values
    return out


def add_post_image_paths(table: pd.DataFrame, data_dir: Path | str) -> pd.DataFrame:
    media = Path(data_dir) / "media"
    wanted = set(zip(table["author"], table["post"]))
    attached = {}
    with (media / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["author"], record["post"])
            if (
                key in wanted and record["kind"] == "post_image"
                and record.get("from_reshared_post") is False
                and (record.get("content_type") or "").startswith("image/")
            ):
                attached.setdefault(key, set()).add(str((media / record["file"]).resolve()))
    out = table.copy()
    out["post_images"] = [
        tuple(sorted(attached.get(key, ()))) for key in zip(out["author"], out["post"])
    ]
    return out


def load_base_dataset(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    posts = author_level(load_posts(data_dir))
    built = features.build(posts, load_followers(data_dir))
    built["age_days"] = (
        built["crawled_at"] - built["posted_at"]
    ).dt.total_seconds() / 86400
    built = add_author_context(built)
    table = load_labels(data_dir).merge(
        built, on=["author", "post"], how="left", validate="one_to_one"
    )
    if table["level"].isna().any():
        raise ValueError("Some labelled posts have no author history; check the release.")
    table["text"] = table["text"].fillna("").str.replace(
        r"⟨(PERSON|ORG|AUTHOR)_\w+⟩", r"⟨\1⟩", regex=True
    )
    table = add_style_features(table)
    table = add_post_image_paths(table, data_dir)
    table = add_image_features(table)
    columns = ["author", "post", "label"] + INPUTS
    train = table.loc[table["split"] == "train", columns].copy()
    test = table.loc[table["split"] == "test", columns].copy()
    if set(train["author"]) & set(test["author"]):
        raise ValueError("Train and test must contain different authors.")
    return train, test


def ensure_pretrained_model(directory, model_id, revision, files):
    directory = Path(directory)
    if not all((directory / file).is_file() for file in files):
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, revision=revision, local_dir=str(directory),
                          allow_patterns=files, max_workers=2)
    return str(directory)


def frozen_cache_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)

class SentenceEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self, model_dir, revision, cache_dir, batch_size=8):
        self.model_dir = model_dir
        self.revision = revision
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def _path(self, text):
        key = hashlib.sha256(('sentence-minilm-multilingual-v1|' + self.revision + '|' + text).encode()).hexdigest()
        return Path(self.cache_dir) / 'sentence' / f'{key}.npy'

    def transform(self, X):
        texts = [str(value) for value in X]
        missing = list(dict.fromkeys(text for text in texts if text.strip() and not self._path(text).exists()))
        if missing:
            import torch
            from transformers import AutoModel, AutoTokenizer
            model_path = ensure_pretrained_model(
                self.model_dir, 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', self.revision, ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentence_bert_config.json', 'modules.json', '1_Pooling/config.json', 'model.safetensors'],
            )
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
            assert model.config.model_type == 'bert' and tokenizer.num_special_tokens_to_add() == 2
            model.requires_grad_(False)
            pending, owners, lengths = [], [], []
            sums, totals, remaining = {}, {}, {}

            def flush():
                if not pending:
                    return
                inputs = tokenizer.pad(pending, padding=True, return_tensors='pt')
                with torch.inference_mode():
                    hidden = model(**inputs).last_hidden_state
                    mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                values = pooled.cpu().numpy()
                for owner, weight, value in zip(owners, lengths, values):
                    sums[owner] += value * weight
                    totals[owner] += weight
                    remaining[owner] -= 1
                    if remaining[owner] == 0:
                        vector = sums.pop(owner) / totals.pop(owner)
                        norm = np.linalg.norm(vector)
                        if norm:
                            vector /= norm
                        frozen_cache_write(self._path(owner), vector)
                        del remaining[owner]
                pending.clear()
                owners.clear()
                lengths.clear()

            for number, text in enumerate(missing):
                token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
                if not token_ids:
                    frozen_cache_write(self._path(text), np.zeros(384, dtype=np.float32))
                    continue
                sums[text] = np.zeros(384, dtype=np.float32)
                totals[text] = 0
                remaining[text] = (len(token_ids) + 125) // 126
                for start in range(0, len(token_ids), 126):
                    chunk = token_ids[start:start + 126]
                    ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
                    pending.append({'input_ids': ids, 'attention_mask': [1] * len(ids),
                                    'token_type_ids': [0] * len(ids)})
                    owners.append(text)
                    lengths.append(len(chunk))
                    if len(pending) == self.batch_size:
                        flush()
                if (number + 1) % 200 == 0:
                    print(f'Sentence embeddings: {number + 1}/{len(missing)} new texts', flush=True)
            flush()
            assert not sums and not totals and not remaining
            del model
            torch.set_num_threads(previous_threads)
        values = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            if text.strip():
                values[index] = np.load(self._path(text), allow_pickle=False)
        return values

class ClipEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self, model_dir, revision, cache_dir, batch_size=8):
        self.model_dir = model_dir
        self.revision = revision
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def _path(self, image_path):
        path = Path(image_path)
        info = path.stat()
        key = hashlib.sha256(json.dumps(['clip-vit-b32-v1', self.revision, str(path.resolve()),
                                         info.st_size, info.st_mtime_ns]).encode()).hexdigest()
        return Path(self.cache_dir) / 'clip' / f'{key}.npy'

    def transform(self, X):
        posts = [tuple(dict.fromkeys(images)) for images in X]
        paths = list(dict.fromkeys(path for images in posts for path in images))
        missing = [path for path in paths if not self._path(path).exists()]
        if missing:
            import torch
            from transformers import CLIPImageProcessorPil, CLIPVisionModelWithProjection
            model_path = ensure_pretrained_model(
                self.model_dir, 'openai/clip-vit-base-patch32', self.revision, ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'merges.txt', 'vocab.json', 'preprocessor_config.json', 'pytorch_model.bin'],
            )
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            processor = CLIPImageProcessorPil.from_pretrained(model_path, local_files_only=True)
            model = CLIPVisionModelWithProjection.from_pretrained(model_path, local_files_only=True).eval()
            model.requires_grad_(False)
            for start in range(0, len(missing), self.batch_size):
                images, valid_paths = [], []
                for path in missing[start:start + self.batch_size]:
                    try:
                        with Image.open(path) as opened:
                            opened.seek(0)
                            rgba = opened.convert('RGBA')
                            white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                            images.append(Image.alpha_composite(white, rgba).convert('RGB'))
                        valid_paths.append(path)
                    except (OSError, ValueError, UnidentifiedImageError):
                        warnings.warn(f'Unable to read image for CLIP: {path}', RuntimeWarning)
                        frozen_cache_write(self._path(path), np.zeros(512, dtype=np.float32))
                if images:
                    inputs = processor(images=images, return_tensors='pt')
                    for image in images:
                        image.close()
                    with torch.inference_mode():
                        vectors = model(**inputs).image_embeds
                        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
                    for path, vector in zip(valid_paths, vectors.cpu().numpy()):
                        frozen_cache_write(self._path(path), vector)
                if start % (self.batch_size * 25) == 0:
                    print(f'CLIP embeddings: {min(start + self.batch_size, len(missing))}/{len(missing)} new images', flush=True)
            del model
            torch.set_num_threads(previous_threads)
        values = np.zeros((len(posts), 512), dtype=np.float32)
        for index, images in enumerate(posts):
            if images:
                vector = np.mean([np.load(self._path(path), allow_pickle=False) for path in images], axis=0)
                norm = np.linalg.norm(vector)
                if norm:
                    vector /= norm
                values[index] = vector
        return values


def text_encoder():
    return SentenceEmbeddings(
        RESULTS / 'pretrained' / 'sentence',
        'e8f8c211226b894fcb81acc59f3b34ba3efd5f42', RESULTS / 'frozen_cache',
    )


def clip_encoder():
    return ClipEmbeddings(
        RESULTS / 'pretrained' / 'clip',
        '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268', RESULTS / 'frozen_cache',
    )


def svm_model() -> Pipeline:
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("cat", OneHotEncoder(
                handle_unknown="ignore", min_frequency=20, sparse_output=True,
            ), features.CONTENT_CAT),
            ("text", TfidfVectorizer(
                ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=30000,
                sublinear_tf=True, dtype=np.float32,
            ), "combined_text"),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.95,
                max_features=30000, sublinear_tf=True, dtype=np.float32,
            ), "text"),
            ("sentence", text_encoder(), "text"),
            ("clip", clip_encoder(), 'publication_images'),
            ("extended_history", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), HISTORY_COLUMNS + CALENDAR_COLUMNS),
        ], sparse_threshold=1.0, transformer_weights={'text': 3.0, 'char': 0.25, 'sentence': 0.5, 'clip': 2.0, 'extended_history': 1.0})),
        ('sort', SortedCSR()),
        ("clf", LinearSVC(
            C=0.012, class_weight={0: 3, 1: 1, 2: 3}, dual=False,
            max_iter=5000, tol=1e-3, random_state=SEED,
        )),
    ])


MODEL_CONFIG = {'stack_folds': 3, 'knn': {'k': 15},
 'tree': {'iterations': 1032, 'leaves': 31, 'learning_rate': 0.04, 'l2': 5.0},
 'meta': {'C': 0.1, 'offsets': [0.55, 0.0, 0.45]},
 'regression': {'loss': 'squared_error', 'leaves': 63}}
MODEL_DESCRIPTION = 'Author-held-out stacking: svm, regression, tree, cosine kNN'
SELECTION = {
    'validation_protocol': '5 outer author folds; 3 inner base folds. k and offsets selected inside outer TRAIN.',
    'selection_test_used': False,
}

TABULAR_NUMERIC = NUMERIC + HISTORY_COLUMNS + CALENDAR_COLUMNS + PROFILE_NUM + RETRIEVAL_COLUMNS + CONTENT_EXTRA_NUM
TABULAR_CATEGORICAL = features.CONTENT_CAT + PROFILE_CAT
COMBINED_COLUMNS = [f'combined_embedding_{i:03d}' for i in range(384)]
ENSEMBLE_COLUMNS = {
    'tree': {'numeric': TABULAR_NUMERIC + COMBINED_COLUMNS,
             'categorical': TABULAR_CATEGORICAL,
             'ordered': TABULAR_NUMERIC + TABULAR_CATEGORICAL + COMBINED_COLUMNS},
}
SENTENCE_COLUMNS = [f'frozen_sentence_{i:03d}' for i in range(384)]
CLIP_COLUMNS = [f'frozen_clip_{i:03d}' for i in range(512)]
ENSEMBLE_COLUMNS['regression'] = {'numeric': TABULAR_NUMERIC + SENTENCE_COLUMNS + CLIP_COLUMNS,
                                  'categorical': TABULAR_CATEGORICAL}
DENSE_INPUTS = list(dict.fromkeys(
    [column for specification in ENSEMBLE_COLUMNS.values()
     for column in specification['numeric'] + specification['categorical']]
))
ADDITIONAL_INPUTS = DENSE_INPUTS + ['combined_text', 'embedding_text', 'publication_images']
MODEL_INPUTS = ["author", "post"] + DENSE_INPUTS + ["text", "combined_text", "publication_images"]


def model():
    return AuthorStackingClassifier(svm_model, MODEL_CONFIG, ENSEMBLE_COLUMNS, COMBINED_COLUMNS)


def load_dataset(data_dir):
    train, test = load_base_dataset(data_dir)
    encoder = text_encoder()
    image_encoder = clip_encoder()
    train = augment(train, data_dir, ADDITIONAL_INPUTS, encoder, regression_target=True, image_encoder=image_encoder)
    test = augment(test, data_dir, ADDITIONAL_INPUTS, encoder, image_encoder=image_encoder)
    return train, test


def main():
    parser = argparse.ArgumentParser(description='Train the fixed author-held-out model and evaluate it.')
    parser.add_argument('--data', default=DATA_DIR, help='unpacked release (default: data/)')
    args = parser.parse_args()
    train, test = load_dataset(args.data)
    print(f'train {len(train):,} posts / {train.author.nunique():,} authors; '
          f'test {len(test):,} posts / {test.author.nunique():,} authors', flush=True)
    estimator = model().fit(train[MODEL_INPUTS], train.label, residual_target=train.get('_residual_target'))
    predicted = estimator.predict(test[MODEL_INPUTS])
    test_metrics = score(test.label, predicted)
    print(f'test: {test_metrics}', flush=True)
    (RESULTS / 'predictions').mkdir(parents=True, exist_ok=True)
    test[['author', 'post']].assign(label=predicted).to_csv(RESULTS / 'predictions/my_model.csv', index=False)
    report = {
        'split': 'test', 'seed': SEED, 'scikit_learn': sklearn.__version__,
        'model': MODEL_DESCRIPTION, 'parameters': MODEL_CONFIG,
        'feature_columns': ENSEMBLE_COLUMNS, 'n_train': len(train), 'n_test': len(test),
        'selection': SELECTION,
        'training': {'combining_model': '3-fold author OOF stacking; predictions average the three base models.',
                     'history': 'Release baseline level/history are retained. Added author-context features use strictly earlier timestamp blocks, including initial unlabelled posts.',
                     'snapshot_note': 'Earlier reaction counts and static profiles follow the release baseline convention.'},
        'results': test_metrics,
    }
    (RESULTS / 'my_model.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Wrote results/my_model.json and results/predictions/my_model.csv', flush=True)


if __name__ == '__main__':
    main()

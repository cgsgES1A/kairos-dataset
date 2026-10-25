import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from kairos.data import load_posts


WINDOWS = (3, 5, 10, 20)
HISTORY_COLUMNS = ['past_count', 'past_last_relative', 'past_mean_relative',
                   'past_q10_relative', 'past_q90_relative', 'past_zero_fraction',
                   'past_ewm3_relative', 'past_ewm10_relative',
                   'past_same_type_count', 'past_same_type_mean_relative',
                   'past_same_type_std', 'past_recent_change', 'past_slope10']
for _window in WINDOWS:
    HISTORY_COLUMNS += [f'past_{_window}_{stat}' for stat in
                        ('mean_relative', 'median_relative', 'std', 'min_relative',
                         'max_relative', 'below_fraction', 'above_fraction')]
CALENDAR_COLUMNS = ['publication_year', 'publication_month', 'hour_sin', 'hour_cos',
                    'weekday_sin', 'weekday_cos']


def add_history_features(posts):
    out = posts.copy()
    values = pd.DataFrame(np.nan, index=out.index, columns=HISTORY_COLUMNS)
    dated = out.loc[out.posted_at.notna()].sort_values(['author', 'posted_at'], kind='mergesort')
    for _, group in dated.groupby('author', sort=False):
        logs = np.log1p(group.reactions.clip(lower=0).to_numpy(dtype=float))
        types = group.content_type.to_numpy()
        dates = group.posted_at.to_numpy()
        starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
        ends = np.r_[starts[1:], len(group)]
        for start, end in zip(starts, ends):
            if not start:
                continue
            past = logs[:start]
            level = float(np.median(past))
            row = {'past_count': float(start), 'past_last_relative': float(past[-1] - level),
                   'past_mean_relative': float(past.mean() - level),
                   'past_q10_relative': float(np.quantile(past, .1) - level),
                   'past_q90_relative': float(np.quantile(past, .9) - level),
                   'past_zero_fraction': float(np.mean(past == 0))}
            for span in (3, 10):
                decay = 1. - 2. / (span + 1.)
                weights = decay ** np.arange(start - 1, -1, -1)
                row[f'past_ewm{span}_relative'] = float(np.average(past, weights=weights) - level)
            for window in WINDOWS:
                recent = past[-window:]
                row.update({f'past_{window}_mean_relative': float(recent.mean() - level),
                            f'past_{window}_median_relative': float(np.median(recent) - level),
                            f'past_{window}_std': float(recent.std()),
                            f'past_{window}_min_relative': float(recent.min() - level),
                            f'past_{window}_max_relative': float(recent.max() - level),
                            f'past_{window}_below_fraction': float(np.mean(recent - level < -.549)),
                            f'past_{window}_above_fraction': float(np.mean(recent - level > 1.157))})
            if start >= 6:
                row['past_recent_change'] = float(past[-3:].mean() - past[-min(10, start):-3].mean())
            if start >= 3:
                recent = past[-10:]
                x = np.arange(len(recent), dtype=float)
                x -= x.mean()
                row['past_slope10'] = float(np.dot(x, recent) / np.dot(x, x))
            for position in range(start, end):
                matched = past[types[:start] == types[position]]
                current = dict(row, past_same_type_count=float(len(matched)),
                               past_same_type_mean_relative=float(matched.mean() - level) if len(matched) else np.nan,
                               past_same_type_std=float(matched.std()) if len(matched) else np.nan)
                values.loc[group.index[position]] = [current.get(name, np.nan) for name in HISTORY_COLUMNS]
    out[HISTORY_COLUMNS] = values
    out['publication_year'] = out.posted_at.dt.year
    out['publication_month'] = out.posted_at.dt.month
    hour = out.posted_at.dt.hour
    weekday = out.posted_at.dt.weekday
    out['hour_sin'], out['hour_cos'] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    out['weekday_sin'], out['weekday_cos'] = np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)
    return out


PROFILE_NUM = ['profile_skills', 'profile_jobs', 'profile_educations', 'profile_languages',
               'profile_certifications', 'profile_summary_words', 'profile_headline_words',
               'profile_is_creator', 'profile_is_premium', 'profile_years_career',
               'profile_jobs_started', 'profile_active_jobs', 'profile_word_overlap', 'profile_skill_overlap']
PROFILE_CAT = ['profile_language', 'profile_country', 'profile_industry']
_TOKEN = re.compile(r'(?u)\b\w\w+\b')


def tokens(text):
    return set(_TOKEN.findall(re.sub(r'⟨[^⟩]*⟩', ' ', str(text)).lower()))


def profile_records(data_dir, authors):
    profiles = {}
    for author in sorted(set(authors)):
        d = json.loads((Path(data_dir) / 'authors' / f'{author}.json').read_text(encoding='utf-8'))
        skills = [str(s.get('name') or '') for s in d.get('skills') or []]
        positions = d.get('fullPositions') or []
        current = d.get('position') or {}
        if isinstance(current, list):
            current = current[0] if current else {}
        text = '\n'.join([str(d.get('headline') or ''), str(d.get('summary') or ''),
                           'Skills: ' + ', '.join(skills),
                           'Industry: ' + str(current.get('companyIndustry') or '')])
        text = re.sub(r'⟨(PERSON|ORG|AUTHOR)_\w+⟩', r'⟨\1⟩', text)
        starts, ends = [], []
        for position in positions:
            start, end = position.get('start') or {}, position.get('end') or {}
            if isinstance(start, dict) and 1900 <= int(start.get('year') or 0) <= 2100:
                starts.append(float(start['year']) + (max(1, int(start.get('month') or 1))-1)/12)
                endyear = int(end.get('year') or 0) if isinstance(end, dict) else 0
                ends.append(float(endyear) + (max(1, int(end.get('month') or 1))-1)/12 if endyear else np.inf)
        profiles[author] = {'text': text, 'tokens': tokens(text), 'skill_tokens': tokens(' '.join(skills)),
            'starts': np.asarray(starts), 'ends': np.asarray(ends),
            'profile_skills': len(skills), 'profile_jobs': len(positions),
            'profile_educations': len(d.get('educations') or []),
            'profile_languages': len(d.get('languages') or []),
            'profile_certifications': len(d.get('certifications') or []),
            'profile_summary_words': len(str(d.get('summary') or '').split()),
            'profile_headline_words': len(str(d.get('headline') or '').split()),
            'profile_is_creator': float(bool(d.get('isCreator'))),
            'profile_is_premium': float(bool(d.get('isPremium'))),
            'profile_language': str((d.get('primaryLocale') or {}).get('language') or 'unknown'),
            'profile_country': str((d.get('geo') or {}).get('countryCode') or 'unknown'),
            'profile_industry': str(current.get('companyIndustry') or 'unknown')}
    return profiles


def add_profile_features(posts, profiles):
    rows = []
    for row in posts.itertuples():
        p = profiles[row.author]
        values = {name: p[name] for name in PROFILE_NUM[:9] + PROFILE_CAT}
        year = row.posted_at.year + (row.posted_at.month-1)/12 if pd.notna(row.posted_at) else np.nan
        started = p['starts'][p['starts'] <= year]
        values['profile_years_career'] = year-started.min() if len(started) else np.nan
        values['profile_jobs_started'] = len(started) if np.isfinite(year) else np.nan
        values['profile_active_jobs'] = int(np.sum((p['starts'] <= year) & (p['ends'] >= year))) if np.isfinite(year) else np.nan
        terms = tokens(row.text)
        values['profile_word_overlap'] = len(terms & p['tokens']) / max(1, len(terms))
        values['profile_skill_overlap'] = len(terms & p['skill_tokens']) / max(1, len(terms))
        rows.append({'author': row.author, 'post': row.post, **values})
    return pd.DataFrame(rows)


CONTENT_EXTRA_NUM = ['attachment_image_count', 'attachment_has_article', 'attachment_has_video',
    'attachment_has_document', 'attachment_document_pages', 'attachment_has_reshare',
    'attachment_reshare_words', 'attachment_extra_words', 'attachment_has_celebration',
    'attachment_has_poll', 'attachment_company_mentions', 'attachment_person_mentions']


def extra_content(data_dir, keys):
    wanted = set(map(tuple, keys[['author', 'post']].itertuples(index=False, name=None)))
    rows=[]
    for author in sorted(keys.author.unique()):
        feed=json.loads((Path(data_dir)/'posts'/f'{author}.json').read_text(encoding='utf-8'))
        for p in feed['posts']:
            if (author, p.get('urn')) not in wanted:
                continue
            article=p.get('article') or {}
            entity=p.get('entity') or {}
            document=p.get('document') or {}
            reshare=p.get('resharedPost') or {}
            celebration=p.get('celebration') or {}
            poll=p.get('poll') or {}
            texts=[]
            for source, fields in [(article, ['title', 'subtitle']), (entity, ['title', 'subtitle']),
                                  (document, ['title', 'transcript']), (reshare, ['text']),
                                  (celebration, ['text']), (poll, ['title'])]:
                for field in fields:
                    value=source.get(field)
                    if isinstance(value, str) and value.strip() and value not in texts:
                        texts.append(value)
            text=re.sub(r'⟨(PERSON|ORG|AUTHOR)_\w+⟩', r'⟨\1⟩', '\n'.join(texts))
            images=p.get('images') or []
            rows.append({'author':author, 'post':p['urn'], 'extra_text':text,
                'attachment_image_count':len(images) if isinstance(images, list) else int(bool(images)),
                'attachment_has_article':int(bool(article)), 'attachment_has_video':int(bool(p.get('video'))),
                'attachment_has_document':int(bool(document)),
                'attachment_document_pages':int(document.get('totalPageCount') or 0),
                'attachment_has_reshare':int(bool(reshare)),
                'attachment_reshare_words':len(str(reshare.get('text') or '').split()),
                'attachment_extra_words':len(text.split()),
                'attachment_has_celebration':int(bool(celebration)),
                'attachment_has_poll':int(bool(poll)),
                'attachment_company_mentions':len(p.get('companyMentions') or []),
                'attachment_person_mentions':len(p.get('mentions') or [])})
    return pd.DataFrame(rows)


K_VALUES = (3, 5, 10)

RETRIEVAL_STATS = ('mean_relative', 'weighted_relative', 'std', 'below_fraction',
                   'above_fraction', 'similarity')
RETRIEVAL_COLUMNS = [column for prefix in ('bm25', 'semantic')
                     for column in ([prefix + '_count', prefix + '_max'] +
                                    [f'{prefix}_{k}_{stat}' for k in K_VALUES for stat in RETRIEVAL_STATS])]
RETRIEVAL_COLUMNS += ['semantic_profile_all', 'semantic_profile_recent5', 'semantic_author_profile']


def word_counts(text):
    text = re.sub(r'⟨[^⟩]*⟩', ' ', str(text)).lower()
    return Counter(t for t in _TOKEN.findall(text) if t not in ENGLISH_STOP_WORDS)


def summarize(scores, logs, level, prefix):
    eligible = np.flatnonzero(scores > 0)
    result = {prefix+'_count': float(len(eligible)), prefix+'_max': float(scores.max()) if len(scores) else np.nan}
    order = eligible[np.argsort(-scores[eligible], kind='stable')]
    for k in K_VALUES:
        selected = order[:k]
        if len(selected):
            residual = logs[selected] - level
            weights = scores[selected]
            values = [float(residual.mean()), float(np.average(residual, weights=weights)),
                      float(residual.std()), float(np.mean(residual < -.549)),
                      float(np.mean(residual > 1.157)), float(scores[selected].mean())]
        else:
            values = [np.nan]*6
        for stat, value in zip(RETRIEVAL_STATS, values):
            result[f'{prefix}_{k}_{stat}'] = value
    return result


def history_retrieval(posts, embeddings=None):
    work = posts.copy()
    work['_position'] = np.arange(len(work))
    work = work.loc[work.posted_at.notna()].sort_values(['author', 'posted_at'], kind='mergesort')
    output = []
    for author, group in work.groupby('author', sort=False):
        texts = [word_counts(value) for value in group.text]
        lengths = np.asarray([sum(c.values()) for c in texts], dtype=float)
        logs = np.log1p(group.reactions.clip(lower=0).to_numpy(float))
        times = group.posted_at.to_numpy()
        start_positions = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
        end_positions = np.r_[start_positions[1:], len(group)]
        inverted = defaultdict(list)
        vectors = embeddings[group['_position'].to_numpy()] if embeddings is not None else None
        for start, end in zip(start_positions, end_positions):
            level = float(np.median(logs[:start])) if start else np.nan
            for position in range(start, end):
                result = {'author': author, 'post': group.iloc[position]['post']}
                scores = np.zeros(start, dtype=float)
                if start:
                    average_length = max(1., float(lengths[:start].mean()))
                    denominator = 1.2*(1.-.75 + .75*lengths[:start]/average_length)
                    for term in texts[position]:
                        occurrences = inverted.get(term)
                        if not occurrences:
                            continue
                        pairs = np.asarray(occurrences)
                        indices, counts = pairs[:, 0].astype(int), pairs[:, 1]
                        idf = np.log1p((start-len(indices)+.5)/(len(indices)+.5))
                        scores[indices] += idf * counts*2.2/(counts+denominator[indices])
                result.update(summarize(scores, logs[:start], level, 'bm25'))
                if vectors is not None:
                    dense = vectors[:start] @ vectors[position]
                    result.update(summarize(dense, logs[:start], level, 'semantic'))
                    for window, name in [(start, 'all'), (min(5, start), 'recent5')]:
                        if window:
                            center = vectors[start-window:start].mean(0)
                            norm = np.linalg.norm(center)
                            value = float(center @ vectors[position] / norm) if norm else np.nan
                        else:
                            value = np.nan
                        result['semantic_profile_'+name] = value
                output.append(result)
            for position in range(start, end):
                for term, count in texts[position].items():
                    inverted[term].append((position, count))
    return pd.DataFrame(output)


def publication_images(data_dir, keys):
    media=Path(data_dir)/'media'
    keys=list(keys)
    wanted=set(map(tuple, keys))
    allowed={'post_image', 'article_image', 'video_poster', 'document_cover', 'document_page', 'celebration_image'}
    attached={}
    with (media/'manifest.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record=json.loads(line)
            key=(record['author'], record['post'])
            if key in wanted and record['kind'] in allowed and (record.get('content_type') or '').startswith('image/'):
                attached.setdefault(key, set()).add(record['file'])
    resolved={}
    for file in {file for group in attached.values() for file in group}:
        path=(media/file).resolve()
        if path.is_file():
            resolved[file]=str(path)
    return [tuple(sorted({resolved[file] for file in attached.get(tuple(key), ()) if file in resolved})) for key in keys]


def augment(table, data_dir, needed, encoder=None, regression_target=False, image_encoder=None):
    needed=set(needed)
    out=table.reset_index(drop=True).copy()
    posts=load_posts(data_dir)
    posts=posts.loc[posts.author.isin(out.author.unique())].copy()
    keys=out[['author', 'post']]
    embeddings=None
    if needed&set(HISTORY_COLUMNS+CALENDAR_COLUMNS):
        print('Preparing earlier-post statistics...', flush=True)
        history=add_history_features(posts)
        out=out.merge(history[['author', 'post']+HISTORY_COLUMNS+CALENDAR_COLUMNS], on=['author', 'post'], how='left', validate='one_to_one')
    if needed&set(PROFILE_NUM+PROFILE_CAT) or 'semantic_author_profile' in needed:
        profiles=profile_records(data_dir, out.author)
        if needed&set(PROFILE_NUM+PROFILE_CAT):
            out=out.merge(add_profile_features(posts, profiles), on=['author', 'post'], how='left', validate='one_to_one')
    if needed&set(CONTENT_EXTRA_NUM+['extra_text', 'combined_text']):
        out=out.merge(extra_content(data_dir, keys), on=['author', 'post'], how='left', validate='one_to_one')
        out.extra_text=out.extra_text.fillna('')
        out['combined_text']=out.text+'\n'+out.extra_text
        out['embedding_text']=out.combined_text.where(out.extra_text.str.strip().ne(''), out.text)
    if any(name.startswith(('bm25_', 'semantic_')) for name in needed):
        dense=any(name.startswith('semantic_') for name in needed)
        if dense:
            if encoder is None:
                raise ValueError('A frozen text encoder is required by semantic history features.')
            normalized=posts.text.fillna('').str.replace(r'⟨(PERSON|ORG|AUTHOR)_\w+⟩', r'⟨\1⟩', regex=True)
            print('Preparing frozen history embeddings...', flush=True)
            embeddings=encoder.transform(normalized)
            positions=pd.Series(np.arange(len(posts)), index=pd.MultiIndex.from_frame(posts[['author', 'post']]))
        print('Retrieving similar strictly earlier posts...', flush=True)
        out=out.merge(history_retrieval(posts, embeddings), on=['author', 'post'], how='left', validate='one_to_one')
        if 'semantic_author_profile' in needed:
            authors=sorted(profiles)
            vectors=encoder.transform([profiles[a]['text'] for a in authors])
            author_positions=pd.Series(np.arange(len(authors)), index=authors)
            current=embeddings[positions.loc[pd.MultiIndex.from_frame(keys)].to_numpy()]
            matched=vectors[author_positions.loc[out.author].to_numpy()]
            out['semantic_author_profile']=np.sum(current*matched, axis=1)
    if 'publication_images' in needed:
        out['publication_images']=publication_images(data_dir, list(zip(out.author, out.post)))
    for branch, dimension in [('sentence', 384), ('clip', 512)]:
        columns=[f'frozen_{branch}_{i:03d}' for i in range(dimension)]
        if needed&set(columns):
            transform=encoder if branch=='sentence' else image_encoder
            if transform is None:
                raise ValueError(f'A frozen {branch} encoder is required.')
            if branch=='sentence' and embeddings is not None:
                values=embeddings[positions.loc[pd.MultiIndex.from_frame(out[['author', 'post']])].to_numpy()]
            else:
                values=transform.transform(out.text if branch=='sentence' else out.post_images)
            out=pd.concat([out, pd.DataFrame(values, index=out.index, columns=columns)], axis=1)
    combined_columns=[f'combined_embedding_{i:03d}' for i in range(384)]
    if needed&set(combined_columns):
        if encoder is None:
            raise ValueError('A frozen text encoder is required.')
        values=encoder.transform(out.embedding_text)
        out=pd.concat([out, pd.DataFrame(values, index=out.index, columns=combined_columns)], axis=1)
    if regression_target:
        reactions=posts.set_index(['author', 'post']).reactions.reindex(pd.MultiIndex.from_frame(keys)).to_numpy(float)
        out['_residual_target']=np.log1p(np.maximum(0, reactions))-out.level.to_numpy()
    missing=needed-set(out.columns)
    if missing:
        raise ValueError(f'Missing required model inputs: {sorted(missing)}')
    return out

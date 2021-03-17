# Datasets

## Summary

4 datasets, from 3 different sources, 2 datasets comprised of
Python functions and 2 comprised of Java functions.

- `c2s/java-small` (from `code2seq`'s evaluation)
- `csn/java` (from GitHub's `CodeSearchNet`) 
- `csn/python` (from GitHub's `CodeSearchNet`)
- `sri/py150` (from SRI Lab's `py150k` dataset)

## Raw Format

Already, at this stage, things are normalized into a `jsonl.gz` encoding. That is,
each file is `gzip` compressed and each line of the file is a `json` object with the
following keys: `granularity`, `code`, `language`.

The key problem is, at this stage, some data points are of `file` granularity whereas
other data points are already at the `method` granularity (which is what we desire
downstream).

From scratch, raw datasets are generated on a decently powerful workstation in under ten minutes. (This mostly depends on download speeds.)

```
Raw Datasets:

  + c2s/java-small/...
    - '/mnt/c2s/java-small/test.jsonl.gz' (7.1M)
    - '/mnt/c2s/java-small/train.jsonl.gz' (69M)
    - '/mnt/c2s/java-small/valid.jsonl.gz' (1.9M)

  + csn/java/...
    - '/mnt/csn/java/test.jsonl.gz' (3.7M)
    - '/mnt/csn/java/train.jsonl.gz' (60M)
    - '/mnt/csn/java/valid.jsonl.gz' (1.8M)

  + csn/python/...
    - '/mnt/csn/python/test.jsonl.gz' (5.4M)
    - '/mnt/csn/python/train.jsonl.gz' (97M)
    - '/mnt/csn/python/valid.jsonl.gz' (5.7M)

  + sri/py150/...
    - '/mnt/sri/py150/test.jsonl.gz' (24M)
    - '/mnt/sri/py150/train.jsonl.gz' (190M)
    - '/mnt/sri/py150/valid.jsonl.gz' (23M)
```

## Normalized Format

Here, everything is at the `method` granularity. Datasets are also trimmed and re-sampled (although sampling is kept within original train/test/valid splits). Things like parsability for Java/Python are enforced. Things like `abstract` methods
in Java are filtered out. The normalizer operates in parallel and can take advantage of a large workstations. The files are kept in the `.jsonl.gz` encoding.

From the raw datasets, normalization takes about an hour on a decently powerful workstation machine. (32 core / 64 thread, 256GB mem.)

```
Normalized Datasets:

  + c2s/java-small/...
    - '/mnt/c2s/java-small/test.jsonl.gz' (6.8M)
    - '/mnt/c2s/java-small/train.jsonl.gz' (19M)
    - '/mnt/c2s/java-small/valid.jsonl.gz' (2.1M)
  
  + csn/java/...
    - '/mnt/csn/java/test.jsonl.gz' (7.9M)
    - '/mnt/csn/java/train.jsonl.gz' (28M)
    - '/mnt/csn/java/valid.jsonl.gz' (3.6M)

  + csn/python/...
    - '/mnt/csn/python/test.jsonl.gz' (12M)
    - '/mnt/csn/python/train.jsonl.gz' (39M)
    - '/mnt/csn/python/valid.jsonl.gz' (5.9M)

  + sri/py150/...
    - '/mnt/sri/py150/test.jsonl.gz' (6.7M)
    - '/mnt/sri/py150/train.jsonl.gz' (24M)
    - '/mnt/sri/py150/valid.jsonl.gz' (3.3M)
```

***Note:*** These numbers are for normalized datasets trimmed to `70,000` samples for train, `10,000` samples for valid, and `20,000` samples for test.

## Preprocessed Format

Here, we pre-process datasets for the two different models we use (`code2seq` and a `seq2seq` baseline model).

### Tokens Representation (for seq2seq)

```
Preprocessed Datasets (tokens):

  + c2s/java-small/...
    - '/mnt/c2s/java-small/test.tsv' (11M)
    - '/mnt/c2s/java-small/train.tsv' (27M)
    - '/mnt/c2s/java-small/valid.tsv' (2.7M)

  + csn/java/...
    - '/mnt/csn/java/test.tsv' (12M)
    - '/mnt/csn/java/train.tsv' (41M)
    - '/mnt/csn/java/valid.tsv' (4.9M)

  + csn/python/...
    - '/mnt/csn/python/test.tsv' (17M)
    - '/mnt/csn/python/train.tsv' (57M)
    - '/mnt/csn/python/valid.tsv' (8.6M)

  + sri/py150/...
    - '/mnt/sri/py150/test.tsv' (8.7M)
    - '/mnt/sri/py150/train.tsv' (30M)
    - '/mnt/sri/py150/valid.tsv' (4.2M)
```

### AST Paths Representation (for code2seq)

```
Preprocessed Datasets (ast-paths):

  + c2s/java-small/...
    - '/mnt/c2s/java-small/data.dict.c2s' (408K)
    - '/mnt/c2s/java-small/data.test.c2s' (121M)
    - '/mnt/c2s/java-small/data.train.c2s' (634M)
    - '/mnt/c2s/java-small/data.val.c2s' (42M)
    - '/mnt/c2s/java-small/histo.node.c2s' (4.0K)
    - '/mnt/c2s/java-small/histo.ori.c2s' (344K)
    - '/mnt/c2s/java-small/histo.tgt.c2s' (68K)

  + csn/java/...
    - '/mnt/csn/java/data.dict.c2s' (376K)
    - '/mnt/csn/java/data.test.c2s' (146M)
    - '/mnt/csn/java/data.train.c2s' (903M)
    - '/mnt/csn/java/data.val.c2s' (70M)
    - '/mnt/csn/java/histo.node.c2s' (4.0K)
    - '/mnt/csn/java/histo.ori.c2s' (516K)
    - '/mnt/csn/java/histo.tgt.c2s' (92K)

  + csn/python/...
    - '/mnt/csn/python/data.dict.c2s' (364K)
    - '/mnt/csn/python/data.test.c2s' (171M)
    - '/mnt/csn/python/data.train.c2s' (1.1G)
    - '/mnt/csn/python/data.val.c2s' (86M)
    - '/mnt/csn/python/histo.node.c2s' (4.0K)
    - '/mnt/csn/python/histo.ori.c2s' (552K)
    - '/mnt/csn/python/histo.tgt.c2s' (168K)

  + sri/py150/...
    - '/mnt/sri/py150/data.dict.c2s' (364K)
    - '/mnt/sri/py150/data.test.c2s' (119M)
    - '/mnt/sri/py150/data.train.c2s' (678M)
    - '/mnt/sri/py150/data.val.c2s' (60M)
    - '/mnt/sri/py150/histo.node.c2s' (4.0K)
    - '/mnt/sri/py150/histo.ori.c2s' (408K)
    - '/mnt/sri/py150/histo.tgt.c2s' (148K)
```


import re
import os
import gzip
import json
import tqdm
import os.path
import multiprocessing


def camel_case_split(identifier):
  matches = re.finditer(
    '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
    identifier
  )
  return [m.group(0) for m in matches]


def subtokens(in_list):
  good_list = []
  for tok in in_list:
    for subtok in tok.replace('_', ' ').split(' '):
      if subtok.strip() != '':
        good_list.extend(camel_case_split(subtok))
  
  return good_list


def clean_name(in_list):
  return subtokens(in_list)


def normalize_subtoken(subtoken):
  normalized = re.sub(
    r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
    re.sub(
      r'["\',`]', r'',     # Get rid of quotes and comma 
      re.sub(
        r'\s+', r'',       # Get rid of spaces
        subtoken.lower()
          .replace('\\\n', '')
          .replace('\\\t', '')
          .replace('\\\r', '')
      )
    )
  )

  return normalized.strip()


def process(item):
  src = list(filter(None, [
    normalize_subtoken(subtok) for subtok in subtokens(item[2])
  ]))
  tgt = list(filter(None, [
    normalize_subtoken(subtok) for subtok in clean_name(item[3])
  ]))

  return (
    len(src) > 0 and len(tgt) > 0,
    item[0],
    item[1],
    ' '.join(src),
    ' '.join(tgt)
  )


if __name__ == "__main__":
  print("Loading inputs...")

  has_baselines = False

  tasks = []
  for split in ["test","train","valid"]:
    if not os.path.isfile('/mnt/inputs/{}.jsonl.gz'.format(split)):
        continue
    if split == 'baseline':
      has_baselines = True
    get_site_map = False
    if os.path.exists('/mnt/inputs/{}_site_map.json'.format(split)): 
      with open('/mnt/inputs/{}_site_map.json'.format(split), 'r') as f:
          site_map = json.load(f)
          print(len(site_map))
      get_site_map = True

    new_site_map = {}
    
    for line in gzip.open('/mnt/inputs/{}.jsonl.gz'.format(split)):
      as_json = json.loads(line)
      from_file = as_json['from_file'] if 'from_file' in as_json else '{}.java'.format(as_json['sha256_hash'])
      from_file = from_file.replace('.py', '')
      from_file = from_file.replace('.java', '')
      tasks.append((split, from_file, as_json['source_tokens'], as_json['target_tokens']))
      the_hash = as_json['sha256_hash']
      if get_site_map:
        new_site_map[from_file] = {}
        # if from_file in site_map:
        for r in site_map[from_file]:
          if site_map[from_file][r][0] == '':
            new_site_map[from_file][r] = site_map[from_file][r]
          else:
            new_site_map[from_file][r] = (' '.join([normalize_subtoken(subtok) for subtok in subtokens([site_map[from_file][r][0]])]), site_map[from_file][r][1])
    
    if get_site_map:
      with open('/mnt/outputs/{}_site_map.json'.format(split), 'w') as f:
          json.dump(new_site_map, f)
  
  pool = multiprocessing.Pool()
  print("  + Inputs loaded")

  out_map = {
    'test': open('/mnt/outputs/test.tsv', 'w'),
    'train': open('/mnt/outputs/train.tsv', 'w'),
    'valid': open('/mnt/outputs/valid.tsv', 'w')
  }

  if has_baselines:
    print("  + Has baselines file")
    out_map['baseline'] = open('/mnt/outputs/baseline.tsv', 'w')
    out_map['baseline'].write('from_file\tsrc\ttgt\n')
  
  print("  + Output files opened")

  out_map['test'].write('from_file\tsrc\ttgt\n')
  out_map['train'].write('from_file\tsrc\ttgt\n')
  out_map['valid'].write('from_file\tsrc\ttgt\n')

  print("  - Processing in parallel...")
  iterator =  tqdm.tqdm(
    pool.imap_unordered(process, tasks, 1000),
    desc="    - Tokenizing",
    total=len(tasks)
  )
  for good, split, from_file, src, tgt in iterator:
    if not good: # Don't let length == 0 stuff slip through
      continue
    out_map[split].write(
      '{}\t{}\t{}\n'.format(from_file, src, tgt)
    )
  print("    + Tokenizing complete")
  print("  + Done extracting tokens")
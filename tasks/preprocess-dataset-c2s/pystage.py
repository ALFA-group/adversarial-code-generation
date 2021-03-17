import sys
import json
import tqdm
import multiprocessing

from fissix import pygram, pytree
from fissix.pgen2 import driver, token


def parse_file(raw_blob):
  try:
    as_json = json.loads(raw_blob)
    contents = as_json['source_code'] + '\n'
    from_file = as_json['from_file']

    parser = driver.Driver(pygram.python_grammar, convert=pytree.convert)

    names_map = token.tok_name
    for key, value in pygram.python_grammar.symbol2number.items():
      names_map[value] = key

    the_ast = parser.parse_string(contents)
    flattened_json = []

    def _traverse(node):
      cur_idx = len(flattened_json)
      if node.type in names_map:
        flattened_json.append({
          'type': names_map[node.type],
          'value': node.value if isinstance(node, pytree.Leaf) else names_map[node.type],
          'children': []
        })
      else:
        assert False, "Type not in map."
      if not isinstance(node, pytree.Leaf):
        for child in node.children:
          flattened_json[cur_idx]["children"].append(_traverse(child))
      return cur_idx

    _traverse(the_ast)

    final_tree = { 
      'from_file': from_file,
      'ast': flattened_json
    }
    
    return json.dumps(
      final_tree,
      separators=(',', ':')
    )
  except Exception as ex:
    return None


if __name__ == "__main__":
  pool = multiprocessing.Pool()
  targets = list(sys.stdin.readlines())
  for result in tqdm.tqdm(pool.imap_unordered(parse_file, targets, 1000), total=len(targets), desc="    - Staging"):
    if result is None:
      continue
    print(result)


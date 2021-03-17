import os
import io
import sys
import ast
import json
import gzip
import copy
import tqdm
import astor
import random
import itertools
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_holes', dest='max_holes', default=50, help='max number of holes to be inserted')
args = parser.parse_args()

PY_2_BUILTINS = [
  'bytearray',
  'IndexError',
  'all',
  'help',
  'vars',
  'SyntaxError',
  'unicode',
  'UnicodeDecodeError',
  'memoryview',
  'isinstance',
  'copyright',
  'NameError',
  'BytesWarning',
  'dict',
  'input',
  'oct',
  'bin',
  'SystemExit',
  'StandardError',
  'format',
  'repr',
  'sorted',
  'False',
  'RuntimeWarning',
  'list',
  'iter',
  'reload',
  'Warning',
  '__package__',
  'round',
  'dir',
  'cmp',
  'set',
  'bytes',
  'reduce',
  'intern',
  'issubclass',
  'Ellipsis',
  'EOFError',
  'locals',
  'BufferError',
  'slice',
  'FloatingPointError',
  'sum',
  'getattr',
  'abs',
  'exit',
  'print',
  'True',
  'FutureWarning',
  'ImportWarning',
  'None',
  'hash',
  'ReferenceError',
  'len',
  'credits',
  'frozenset',
  '__name__',
  'ord',
  'super',
  '_',
  'TypeError',
  'license',
  'KeyboardInterrupt',
  'UserWarning',
  'filter',
  'range',
  'staticmethod',
  'SystemError',
  'BaseException',
  'pow',
  'RuntimeError',
  'float',
  'MemoryError',
  'StopIteration',
  'globals',
  'divmod',
  'enumerate',
  'apply',
  'LookupError',
  'open',
  'quit',
  'basestring',
  'UnicodeError',
  'zip',
  'hex',
  'long',
  'next',
  'ImportError',
  'chr',
  'xrange',
  'type',
  '__doc__',
  'Exception',
  'tuple',
  'UnicodeTranslateError',
  'reversed',
  'UnicodeEncodeError',
  'IOError',
  'hasattr',
  'delattr',
  'setattr',
  'raw_input',
  'SyntaxWarning',
  'compile',
  'ArithmeticError',
  'str',
  'property',
  'GeneratorExit',
  'int',
  '__import__',
  'KeyError',
  'coerce',
  'PendingDeprecationWarning',
  'file',
  'EnvironmentError',
  'unichr',
  'id',
  'OSError',
  'DeprecationWarning',
  'min',
  'UnicodeWarning',
  'execfile',
  'any',
  'complex',
  'bool',
  'ValueError',
  'NotImplemented',
  'map',
  'buffer',
  'max',
  'object',
  'TabError',
  'callable',
  'ZeroDivisionError',
  'eval',
  '__debug__',
  'IndentationError',
  'AssertionError',
  'classmethod',
  'UnboundLocalError',
  'NotImplementedError',
  'AttributeError',
  'OverflowError'
]


def t_rename_fields(the_ast, uid=1, all_sites=False):
	"""
	all_sites=True: a single, randomly selected, referenced field 
	(self.field in Python) has its name replaced by a hole
	all_sites=False: all possible fields are selected
	"""
	changed = False

	# Going to need parent info
	for node in ast.walk(the_ast):
		for child in ast.iter_child_nodes(node):
			child.parent = node

	candidates = []
	for node in ast.walk(the_ast):
		if isinstance(node, ast.Name) and node.id == 'self':
			if isinstance(node.parent, ast.Attribute):
				if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
					continue
				if node.parent.attr not in [ c.attr for c in candidates ]:
					candidates.append(node.parent)

	if len(candidates) == 0:
		return False, the_ast, uid-1, {}

	if not all_sites:
		selected = [random.choice(candidates)]
	else:
		selected = candidates

	to_rename = []
	for cnt, selected_node in enumerate(selected, start=uid):
		for node in ast.walk(the_ast):
			if isinstance(node, ast.Name) and node.id == 'self':
				if isinstance(node.parent, ast.Attribute) and node.parent.attr == selected_node.attr:
					if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
						continue
					to_rename.append((node.parent, cnt))

	site_map = {}
	for node, idx in to_rename:
		changed = True
		site_map["@R_{}@".format(idx)] = (node.attr, "transforms.RenameFields")
		node.attr = 'REPLACEME' + str(idx)

	return changed, the_ast, uid+len(selected)-1, site_map


def t_rename_parameters(the_ast, uid=1, all_sites=False):
	"""
	Parameters get replaced by holes.
	"""
	changed = False
	candidates = []
	for node in ast.walk(the_ast):
		if isinstance(node, ast.arg):
			if node.arg != 'self' and node.arg not in [ c.arg for c in candidates ]:
				# print(node.arg, node.lineno)
				candidates.append(node)

	if len(candidates) == 0:
		return False, the_ast, uid-1, {}

	if not all_sites:
		selected = [random.choice(candidates)]
	else:
		selected = candidates

	parameter_defs = {}
	for cnt, s in enumerate(selected, start=uid):
		parameter_defs[s.arg] = cnt

	to_rename = []
	for node in ast.walk(the_ast):
		if isinstance(node, ast.Name) and node.id in parameter_defs:
			to_rename.append((node, parameter_defs[node.id]))
		elif isinstance(node, ast.arg) and node.arg in parameter_defs:
			to_rename.append((node, parameter_defs[node.arg]))

	site_map = {}
	
	for node, idx in to_rename:
		changed = True
		if hasattr(node, 'arg'):
			site_map["@R_{}@".format(idx)] = (node.arg, "transforms.RenameParameters")
			node.arg = 'REPLACEME' + str(idx)
		else:
			site_map["@R_{}@".format(idx)] = (node.id, "transforms.RenameParameters")
			node.id = 'REPLACEME' + str(idx)

	return changed, the_ast, uid+len(selected)-1, site_map


def t_rename_local_variables(the_ast, uid=1, all_sites=False):
	"""
	Local variables get replaced by holes.
	"""
	changed = False
	candidates = []
	for node in ast.walk(the_ast):
		if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
			if node.id not in [ c.id for c in candidates ]:
				# print(node.id, node.lineno)
				candidates.append(node)

	if len(candidates) == 0:
		return False, the_ast, uid-1, {}

	if not all_sites:
		selected = [random.choice(candidates)]
	else:
		selected = candidates

	local_var_defs = {}
	for cnt, s in enumerate(selected, start=uid):
		local_var_defs[s.id] = cnt

	to_rename = []
	for node in ast.walk(the_ast):
		if isinstance(node, ast.Name) and node.id in local_var_defs:
			to_rename.append((node, local_var_defs[node.id]))

	site_map = {}
	
	for node, idx in to_rename:
		changed = True
		site_map["@R_{}@".format(idx)] = (node.id, "transforms.RenameLocalVariables")
		node.id = 'REPLACEME' + str(idx)

	return changed, the_ast, uid+len(selected)-1, site_map


def t_unroll_whiles(the_ast, uid=1, all_sites=False):
	"""
	While loops in the target program have their loop
	bodies unrolled exactly one step. No holes are created by this transform.
	"""
	if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
		return False, the_ast, uid-1
	
	class UnrollWhiles(ast.NodeTransformer):
		def __init__(self, selection):
			self.selection = selection
			self.count = 0
			self.done = False
			super().__init__()

		def visit_While(self, node):
			if self.done:
				return node
			if self.count != self.selection:
				self.count += 1
				return node
			
			self.done = True
			return ast.While(
				test=node.test,
				body=node.body + [ node, ast.Break() ],
				orelse=[]
			)

	changed = False
	count = 0
	
	for node in ast.walk(the_ast):
		if isinstance(node, ast.While):
			changed = True
			count += 1

	if count == 0:
		return False, the_ast, uid-1, {}

	if all_sites:
		for cnt in range(count):
			the_ast = UnrollWhiles(cnt).visit(the_ast)
	else:
		the_ast = UnrollWhiles(random.randint(0, count - 1)).visit(the_ast)
	
	return changed, the_ast, uid-1, {}


def t_wrap_try_catch(the_ast, uid=1, all_sites=False):
	"""
	The target program is wrapped by a single try { ... } catch (...) {
	... } statement. A hole is used in the place of the name of the caught 
	exception variable (e.g., catch (Exception <HOLE>)).
	"""
	if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
		return False, the_ast, uid-1, {}

	temp = ast.Try(
		body = the_ast.body[0].body,
		handlers=[ast.ExceptHandler(
		type=ast.Name(id='Exception', ctx=ast.Load()),
		name='REPLACEME' + str(uid),
		body=[ast.Raise()]
		)],
		orelse=[],
		finalbody=[]
	)

	the_ast.body[0].body = [temp]
	site_map = {"@R_{}@".format(uid):["", "transforms.WrapTryCatch"]}

	return True, the_ast, uid, site_map


def t_add_dead_code(the_ast, uid=1, all_sites=False):
	"""
	Statement of the form if False:\n <HOLE> = 1 is added to the target program. 
	all_sites=False: The insertion location (either beginning, or end) is chosen at random.
	all_sites=True: The statement is inserted at all possible locations.
	"""
	def insert_holes(the_ast, hole_id, site_map):

		if 'body' not in the_ast.__dict__ or len(the_ast.body) == 0:
			return False, the_ast, hole_id, site_map
		new_body = []
		for node in the_ast.body:
			site_map["if false : @R_{}@ = 1".format(hole_id)] = ["", 'transforms.AddDeadCode']
			new_body += [ast.If(test=ast.Name(id="False", ctx=ast.Load()),
				body=[ast.Assign(
					targets=[ast.Name(id="REPLACEME" + str(hole_id), ctx=ast.Store())],
					value=ast.Num(n=1)
				)], orelse=[]), node]
			hole_id += 1
		site_map["if false : @R_{}@ = 1".format(hole_id)] = ["", 'transforms.AddDeadCode']
		new_body.append(ast.If(test=ast.Name(id="False", ctx=ast.Load()),
				body=[ast.Assign(
					targets=[ast.Name(id="REPLACEME" + str(hole_id), ctx=ast.Store())],
					value=ast.Num(n=1))], orelse=[]))
		hole_id += 1
		the_ast.body = new_body
		
		for i in range(len(the_ast.body[1:])):
			if i%2 == 0:
				node = the_ast.body[i+1]
				_, _, hole_id, site_map = insert_holes(node, hole_id, site_map)
		return True, the_ast, hole_id, site_map

	if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
		return False, the_ast, uid-1, {}

	if all_sites:
		changed, the_ast_body, hole_id, site_map = insert_holes(the_ast.body[0], uid, {})
		the_ast.body[0] = the_ast_body
		return changed, the_ast, hole_id-1, site_map

	else:
		site_map = {}
		if bool(random.getrandbits(1)):
			the_ast.body[0].body.insert(
			0,
			ast.If(
				test=ast.Name(id="False", ctx=ast.Load()),
				body=[ 
				ast.Assign(
					targets=[ast.Name(id="REPLACEME" + str(uid), ctx=ast.Store())],
					value=ast.Num(n=1)
				)
				],
				orelse=[]
			)
			)
			site_map['@R_{}@'.format(uid)] = ["", 'transforms.AddDeadCode']
		else:
			the_ast.body[0].body.append(
			ast.If(
				test=ast.Name(id="False", ctx=ast.Load()),
				body=[ 
				ast.Assign(
					targets=[ast.Name(id="REPLACEME" + str(uid), ctx=ast.Store())],
					value=ast.Num(n=1)
				)
				],
				orelse=[]
			)
			)
			site_map['@R_{}@'.format(uid)] = ["", 'transforms.AddDeadCode']
		
		return True, the_ast, uid, site_map


def t_insert_print_statements(the_ast, uid=1, all_sites=False):
	"""
	Print statements of the form 'print( <HOLE>)' are inserted in the 
	target program. 
	"""

	transform_name = 'transforms.InsertPrintStatements'
	def insert_holes(the_ast, hole_id, site_map):

		if 'body' not in the_ast.__dict__ or len(the_ast.body) == 0:
			return False, the_ast, hole_id, site_map
		new_body = []
		for node in the_ast.body:
			site_map["print ( @R_{}@ )".format(hole_id)] = ["", transform_name]
			new_body += [ast.Expr(ast.Call(
				func=ast.Name(id='print', ctx=ast.Load()),
				args=[ast.Str("REPLACEME{}".format(hole_id))],
				keywords=[])), node]
			hole_id += 1
		site_map["print ( @R_{}@ )".format(hole_id)] = ["", transform_name]
		new_body.append(ast.Expr(ast.Call(
			func=ast.Name(id='print', ctx=ast.Load()),
			args=[ast.Str("REPLACEME{}".format(hole_id))],
			keywords=[])))
		hole_id += 1
		the_ast.body = new_body
		
		for node in the_ast.body:
			_, _, hole_id, site_map = insert_holes(node, hole_id, site_map)
		return True, the_ast, hole_id, site_map

	if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
		return False, the_ast, uid-1, {}

	if all_sites:

		changed, the_ast_body, hole_id, site_map = insert_holes(the_ast.body[0], uid, {})
		the_ast.body[0] = the_ast_body
		return changed, the_ast, hole_id-1, site_map

	else:
		
		site_map = {}
		if bool(random.getrandbits(1)):
			the_ast.body[0].body.insert(
				0,
				ast.Expr(
				ast.Call(
					func=ast.Name(id="print", ctx=ast.Load()),
					args=[ast.Str("REPLACEME" + str(uid))],
					keywords=[]
				)
				)
			)
			site_map['print ( @R_{}@ )'.format(uid)] = ["", transform_name]
		else:
			the_ast.body[0].body.append(
				ast.Expr(
				ast.Call(
					func=ast.Name(id="print", ctx=ast.Load()),
					args=[ast.Str("REPLACEME" + str(uid))],
					keywords=[]
				)
				)
			)
			site_map['print ( @R_{}@ )'.format(uid)] = ["", transform_name]

	
		return True, the_ast, uid, site_map



def t_replace_true_false(the_ast, uid=1, all_sites=False):
	"""
	Boolean literals are replaced by an equivalent
	expression containing a single hole 
	(e.g., ("<HOLE>" == "<HOLE>") to replace true).
	"""
	class ReplaceTrueFalse(ast.NodeTransformer):
		def __init__(self, selection, hole_id):
			self.selection = selection
			self.count = 0
			self.done = False
			self.hole_id = hole_id
			self.node_value = None
			super().__init__()

		def visit_NameConstant(self, node):
			if self.done:
				return node
			if node.value != True and node.value != False:
				return node
			if self.count != self.selection:
				self.count += 1
				return node
			self.done = True
			self.node_value = node.value
			return ast.Compare(
				left=ast.Str("REPLACEME" + str(self.hole_id)),
				ops=[ast.Eq() if node.value == True else ast.NotEq()],
				comparators=[ast.Str("REPLACEME" + str(self.hole_id))]
			)

	changed = False
	count = 0
	
	for node in ast.walk(the_ast):
		if isinstance(node, ast.NameConstant) and node.value == True:
			changed = True
			count += 1
		elif isinstance(node, ast.NameConstant) and node.value == False:
			changed = True
			count += 1

	if count == 0:
		return False, the_ast, uid-1, {}

	site_map = {}
	if all_sites:
		for cnt in range(count):
			repl_true_false = ReplaceTrueFalse(0, cnt+uid)
			the_ast = repl_true_false.visit(the_ast)
			node_value = repl_true_false.node_value
			# the_ast, node_value = ReplaceTrueFalse(0, cnt+uid).visit(the_ast)
			if node_value == True:
				site_map["@R_{}@ == @R_{}@".format(cnt+uid, cnt+uid)] = ['True', 'transforms.ReplaceTrueFalse']
			else:
				assert node_value == False
				site_map["@R_{}@ != @R_{}@".format(cnt+uid, cnt+uid)] = ['False', 'transforms.ReplaceTrueFalse']
		return changed, the_ast, uid+count-1, site_map
	else:
		repl_true_false = ReplaceTrueFalse(random.randint(0, count - 1), uid)
		the_ast = repl_true_false.visit(the_ast)
		node_value = repl_true_false.node_value
		if node_value == True:
			site_map["@R_{}@ == @R_{}@".format(uid, uid)] = ['True', 'transforms.ReplaceTrueFalse']
		else:
			assert node_value == False
			site_map["@R_{}@ != @R_{}@".format(uid, uid)] = ['False', 'transforms.ReplaceTrueFalse']

	
		return changed, the_ast, uid, site_map


class t_seq(object):
	def __init__(self, transforms, all_sites):
		self.transforms = transforms
		self.all_sites = all_sites
	def __call__(self, the_ast, all_sites=False):
		did_change = False
		cur_ast = the_ast
		cur_idx = 0
		new_site_map = {}
		for t in self.transforms:
			changed, cur_ast, cur_idx, site_map = t(cur_ast, cur_idx+1, self.all_sites)
			if changed:
				did_change = True
				new_site_map.update(site_map)
		return did_change, cur_ast, cur_idx, new_site_map


def t_identity(the_ast, all_sites=None):
	return True, the_ast, 0, {}


def process(item):
	(split, the_hash, og_code) = item

	transforms = [
		(
		'transforms.Identity',
		t_identity
		)
	]

	doDepthK = 'DEPTH' in os.environ and len(os.environ['DEPTH']) > 0
	if doDepthK:
		assert 'NUM_SAMPLES' in os.environ and len(os.environ['NUM_SAMPLES']) > 0
		DEPTH = int(os.environ['DEPTH'])
		NUM_SAMPLES = int(os.environ['NUM_SAMPLES'])

		for s in range(NUM_SAMPLES):
			the_seq = []
			for _ in range(DEPTH):
				rand_int = random.randint(1, 8)
				if rand_int == 1:
					the_seq.append(t_replace_true_false)
				elif rand_int == 2:
					the_seq.append(t_rename_local_variables)
				elif rand_int == 3:
					the_seq.append(t_rename_parameters)
				elif rand_int == 4:
					the_seq.append(t_rename_fields)
				elif rand_int == 5:
					the_seq.append(t_insert_print_statements)
				elif rand_int == 6:
					the_seq.append(t_add_dead_code)
				elif rand_int == 7:
					the_seq.append(t_unroll_whiles)
				elif rand_int == 8:
					the_seq.append(t_wrap_try_catch)
		
		transforms.append(('depth-{}-sample-{}'.format(DEPTH, s+1), t_seq(the_seq)))
	else:
		# transforms.append(('renamevar-param', t_seq([t_rename_local_variables, t_rename_parameters], all_sites=True)))
		# transforms.append(('transforms.InsertPrintStatements', t_insert_print_statements))
		# transforms.append(('transforms.RenameLocalVariables',  t_rename_local_variables))
		# transforms.append(('transforms.RenameParameters', t_rename_parameters))		
		# transforms.append(('transforms.ReplaceTrueFalse',  t_replace_true_false))
		# transforms.append(('transforms.RenameFields', t_rename_fields))
		# transforms.append(('transforms.AddDeadCode', t_add_dead_code))
		# transforms.append(('transforms.UnrollWhiles', t_unroll_whiles))
		# transforms.append(('transforms.WrapTryCatch', t_wrap_try_catch))
		transforms.append(('transforms.Combined', t_seq([t_rename_local_variables, t_rename_parameters, t_rename_fields, t_replace_true_false, t_insert_print_statements, t_add_dead_code], all_sites=True)))
		transforms.append(('transforms.Insert', t_seq([t_insert_print_statements, t_add_dead_code], all_sites=True)))
		transforms.append(('transforms.Replace', t_seq([t_rename_local_variables, t_rename_parameters, t_rename_fields, t_replace_true_false], all_sites=True)))
		
	results = []
	for t_name, t_func in transforms:
		try:
			# print(t_func)
			changed, result, last_idx, site_map = t_func(
				ast.parse(og_code),
				all_sites=True
			)
			results.append((changed, split, t_name, the_hash, astor.to_source(result), site_map)) 
		except Exception as ex:
			import traceback
			traceback.print_exc()
			results.append((False, split, t_name, the_hash, og_code, {}))
	return results


if __name__ == "__main__":
	print("Starting transform:")
	pool = multiprocessing.Pool(1)

	tasks = []

	print("  + Loading tasks...")
	splits = ['test', 'train', 'valid']
	if "AVERLOC_JUST_TEST" in os.environ and os.environ['AVERLOC_JUST_TEST'].strip().lower().startswith('t'):
		splits = ['test']

	for split in splits:
		for line in gzip.open('/mnt/inputs/{}.jsonl.gz'.format(split)):
			as_json = json.loads(line)
			the_code = as_json['source_code']
			tasks.append((split, as_json['sha256_hash'], the_code))
	
	print("    + Loaded {} transform tasks".format(len(tasks)))
	# task[114:115] has multiple variables.
	results = pool.imap_unordered(process, tasks, 3000)

	print("  + Transforming in parallel...")
	names_covered = []
	all_sites = {}
	for changed, split, t_name, the_hash, code, site_map in itertools.chain.from_iterable(tqdm.tqdm(results, desc="    + Progress", total=len(tasks))):

		# all_sites[(t_name, split, the_hash)] = site_map
		if t_name not in all_sites:
			all_sites[t_name] = {split:{the_hash:site_map}}

		else:
			if split not in all_sites[t_name]:
				all_sites[t_name][split] = {the_hash:site_map}
			else:
				all_sites[t_name][split][the_hash] = site_map
			
		if (t_name + split) not in names_covered:
			names_covered.append(t_name + split)
			os.makedirs('/mnt/raw-outputs/{}/{}'.format(t_name, split), exist_ok=True)

		with open('/mnt/raw-outputs/{}/{}/{}.py'.format(t_name, split, the_hash), 'w') as fout:
			fout.write('{}\n'.format(code))

	for t_name in all_sites:
		for split in all_sites[t_name]:
			if not os.path.exists('/mnt/outputs/'+t_name):
				os.makedirs('/mnt/outputs/'+t_name)
			with open('/mnt/outputs/{}/{}_site_map.json'.format(t_name, split), 'w') as f:
				json.dump(all_sites[t_name][split], f)
		

	print("  + Transforms complete!")

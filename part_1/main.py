# Copyright (c) 2024 Swastik Majumder
# All rights reserved.

from collections import deque
import copy
import itertools

class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root")
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ')
        node_name = line.strip()
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0]

def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name)
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1)
        return result
    return recursive_str(node)

def apply_individual_axiom(equation, axiom_input, axiom_output, do_only_arithmetic=False):
    variable_list = {}
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def equation_given_satisfy_axiom_structure(equation, axiom):
        nonlocal variable_list
        if node_type(axiom.name) in {"u_", "p_"}:
            if axiom.name in variable_list.keys():
                return str_form(variable_list[axiom.name]) == str_form(equation)
            else:
                if node_type(axiom.name) == "p_" and "v_" in str_form(equation):
                    return False
                variable_list[axiom.name] = copy.deepcopy(equation)
                return True
        if equation.name != axiom.name or len(equation.children) != len(axiom.children):
            return False
        for i in range(len(equation.children)):
            if equation_given_satisfy_axiom_structure(equation.children[i], axiom.children[i]) is False:
                return False
        return True
    def axiom_apply_root(axiom):
        nonlocal variable_list
        if axiom.name in variable_list.keys():
            return variable_list[axiom.name]
        data_to_return = TreeNode(axiom.name, None)
        for child in axiom.children:
            data_to_return.children.append(axiom_apply_root(copy.deepcopy(child)))
        return data_to_return
    count_target_node = 1
    def axiom_apply_various_level(equation, axiom_input, axiom_output, do_only_arithmetic):
        nonlocal variable_list
        nonlocal count_target_node
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if do_only_arithmetic == False:
            if equation_given_satisfy_axiom_structure(equation, copy.deepcopy(axiom_input)) is True:
                count_target_node -= 1
                if count_target_node == 0:
                    return axiom_apply_root(copy.deepcopy(axiom_output))
        else:
            if len(equation.children) == 2 and all(node_type(item.name) == "d_" for item in equation.children):
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:]))
                if equation.name == "f_add":
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_" + str(sum(x)))
                elif equation.name == "f_mul":
                    count_target_node -= 1
                    if count_target_node == 0:
                        p = 1
                        for item in x:
                            p *= item
                        return TreeNode("d_" + str(p))
                elif equation.name == "f_pow" and x[1]>=2:
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
        if node_type(equation.name) in {"d_", "v_"}:
            return equation
        for child in equation.children:
            data_to_return.children.append(axiom_apply_various_level(copy.deepcopy(child), axiom_input, axiom_output, do_only_arithmetic))
        return data_to_return
    cn = 0
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    outputted_val = []
    count_nodes(equation)
    for i in range(1, cn + 1):
        count_target_node = i
        orig_len = len(outputted_val)
        tmp = axiom_apply_various_level(equation, axiom_input, axiom_output, do_only_arithmetic)
        if str_form(tmp) != str_form(equation):
            outputted_val.append(tmp)
    return outputted_val

# Function to read axiom file
def return_axiom_file(file_name):
    content = None
    with open(file_name, 'r') as file:
        content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)]
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f]
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f]

# Function to generate neighbor equations
def generate(node):
    input_f, output_f = return_axiom_file("axiom.txt")
    output_list = []
    output_list += apply_individual_axiom(tree_form(node), None, None, True)
    for i in range(len(input_f)):
        output_list += apply_individual_axiom(tree_form(node), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(output_list))

# Sample input equation to search for
question = """f_add
 f_add
  f_pow
   v_0
   d_2
  d_1
 f_mul
  d_2
  v_0"""

# Function to recursively search for smallest equation
def search(equation, to_find, depth):

    if equation == to_find:
        return True
    if depth == 0:
        return False
    
    output = generate(equation)
    
    for i in range(len(output)):
        #print(output[i])
        result = search(str_form(output[i]), to_find, depth-1)
        if result:
            return True
        
    return False

def fx_nest(terminal, fx, depth):
    def nn(curr_tree, depth=depth):
        def is_terminal(name):
            return not (name in fx.keys())
        element = None
        def append_at_last(curr_node, depth):
            if (is_terminal(element) and depth == 0) or (not is_terminal(element) and depth == 1):
                return None
            if not is_terminal(curr_node.name):
                if len(curr_node.children) < fx[curr_node.name]:
                    curr_node.children.append(TreeNode(element))
                    return curr_node
                for i in range(len(curr_node.children)):
                    output = append_at_last(copy.deepcopy(curr_node.children[i]), depth - 1)
                    if output is not None:
                        curr_node.children[i] = copy.deepcopy(output)
                        return curr_node
            return None
        output = []
        for item in terminal + list(fx.keys()):
            element = item
            tmp = copy.deepcopy(curr_tree)
            result = append_at_last(tmp, depth)
            if result is not None:
                output.append(result)
        return output
    all_poss = []
    def bfs(start_node):
        nonlocal all_poss
        queue = deque()
        visited = set()
        queue.append(start_node)
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                neighbors = nn(current_node)
                if neighbors == []:
                    all_poss.append(str_form(current_node))
                    all_poss = list(set(all_poss))
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
    for item in fx.keys():
        bfs(TreeNode(item))
    return all_poss

def part(term):
    sub_term_list = [term]
    term = tree_form(term)
    for child in term.children:
        sub_term_list += part(str_form(child))
    return sub_term_list

def illegal_eq(term):
    term = tree_form(term)
    if term.name == "f_pow":
        return term.children[1].name[:2] == "d_" and int(term.children[1].name[2:]) >= 2
    return True

leaf = ["d_" + str(i) for i in range(1, 3)] + ["v_" + str(i) for i in range(0, 1)]

con_term = fx_nest(leaf, {"f_add": 2, "f_mul": 2, "f_pow": 2}, 2)

con_term = [term for term in con_term if all(illegal_eq(item) for item in part(term))] + leaf

smallest = "-"*1000
for item in con_term:
    if search(item, question, 4) and len(item) < len(smallest):
        smallest = item
print(smallest)

# Part of the book Transcendental Computing with Python: Applications in Mathematics - Edition 1

from collections import deque
import copy

# Basic data structure, which can nest to represent math equations
class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []

# convert string representation into tree
def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root") # add a dummy node
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(' ') # count the spaces, which is crucial information in a string representation
        node_name = line.strip() # remove spaces, when putting it in the tree form
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0] # remove dummy node

# convert tree into string representation
def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(' ' * depth, node.name) # spacings
        for child in node.children:
            result += "\n" + recursive_str(child, depth + 1) # one node in one line
        return result
    return recursive_str(node)

# Generate transformations of a given equation provided only one formula to do so
# We can call this function multiple times with different formulas, in case we want to use more than one
# This function is also responsible for computing arithmetic, pass do_only_arithmetic as True (others param it would ignore), to do so
def apply_individual_formula_on_given_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic=False):
    variable_list = {}
    def node_type(s):
        if s[:2] == "f_":
            return s
        else:
            return s[:2]
    def does_given_equation_satisfy_forumla_lhs_structure(equation, formula_lhs):
        nonlocal variable_list
        # u can accept anything and p is expecting only integers
        # if there is variable in the formula
        if node_type(formula_lhs.name) in {"u_", "p_"}: 
            if formula_lhs.name in variable_list.keys(): # check if that variable has previously appeared or not
                return str_form(variable_list[formula_lhs.name]) == str_form(equation) # if yes, then the contents should be same
            else: # otherwise, extract the data from the given equation
                if node_type(formula_lhs.name) == "p_" and "v_" in str_form(equation): # if formula has a p type variable, it only accepts integers
                    return False
                variable_list[formula_lhs.name] = copy.deepcopy(equation)
                return True
        if equation.name != formula_lhs.name or len(equation.children) != len(formula_lhs.children): # the formula structure should match with given equation
            return False
        for i in range(len(equation.children)): # go through every children and explore the whole formula / equation
            if does_given_equation_satisfy_forumla_lhs_structure(equation.children[i], formula_lhs.children[i]) is False:
                return False
        return True
    # transform the equation as a whole aka perform the transformation operation on the entire thing and not only on a certain part of the equation
    def formula_apply_root(formula):
        nonlocal variable_list
        if formula.name in variable_list.keys():
            return variable_list[formula.name] # fill the extracted data on the formula rhs structure
        data_to_return = TreeNode(formula.name, None) # produce nodes for the new transformed equation
        for child in formula.children:
            data_to_return.children.append(formula_apply_root(copy.deepcopy(child))) # slowly build the transformed equation
        return data_to_return
    count_target_node = 1
    # try applying formula on various parts of the equation
    def formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic):
        nonlocal variable_list
        nonlocal count_target_node
        data_to_return = TreeNode(equation.name, children=[])
        variable_list = {}
        if do_only_arithmetic == False:
            if does_given_equation_satisfy_forumla_lhs_structure(equation, copy.deepcopy(formula_lhs)) is True: # if formula lhs structure is satisfied by the equation given
                count_target_node -= 1
                if count_target_node == 0: # and its the location we want to do the transformation on
                    return formula_apply_root(copy.deepcopy(formula_rhs)) # transform
        else: # perform arithmetic
            if len(equation.children) == 2 and all(node_type(item.name) == "d_" for item in equation.children): # if only numbers
                x = []
                for item in equation.children:
                    x.append(int(item.name[2:])) # convert string into a number
                if equation.name == "f_add":
                    count_target_node -= 1
                    if count_target_node == 0: # if its the location we want to perform arithmetic on
                        return TreeNode("d_" + str(sum(x))) # add all
                elif equation.name == "f_mul":
                    count_target_node -= 1
                    if count_target_node == 0:
                        p = 1
                        for item in x:
                            p *= item # multiply all
                        return TreeNode("d_" + str(p))
                elif equation.name == "f_pow" and x[1]>=2: # power should be two or a natural number more than two
                    count_target_node -= 1
                    if count_target_node == 0:
                        return TreeNode("d_"+str(int(x[0]**x[1])))
        if node_type(equation.name) in {"d_", "v_"}: # reached a leaf node
            return equation
        for child in equation.children: # slowly build the transformed equation
            data_to_return.children.append(formula_apply_various_sub_equation(copy.deepcopy(child), formula_lhs, formula_rhs, do_only_arithmetic))
        return data_to_return
    cn = 0
    # count how many locations are present in the given equation
    def count_nodes(equation):
        nonlocal cn
        cn += 1
        for child in equation.children:
            count_nodes(child)
    transformed_equation_list = []
    count_nodes(equation)
    for i in range(1, cn + 1): # iterate over all location in the equation tree
        count_target_node = i
        orig_len = len(transformed_equation_list)
        tmp = formula_apply_various_sub_equation(equation, formula_lhs, formula_rhs, do_only_arithmetic)
        if str_form(tmp) != str_form(equation): # don't produce duplication, or don't if nothing changed because of transformation impossbility in that location
            transformed_equation_list.append(tmp) # add this transformation to our list
    return transformed_equation_list 

# Function to read formula file
def return_formula_file(file_name):
    content = None
    with open(file_name, 'r') as file:
        content = file.read()
    x = content.split("\n\n")
    input_f = [x[i] for i in range(0, len(x), 2)] # alternative formula lhs and then formula rhs
    output_f = [x[i] for i in range(1, len(x), 2)]
    input_f = [tree_form(item) for item in input_f] # convert into tree form
    output_f = [tree_form(item) for item in output_f]
    return [input_f, output_f] # return

# Function to generate neighbor equations
def generate_transformation(equation):
    input_f, output_f = return_formula_file("formula_list.txt") # load formula file
    transformed_equation_list = []
    transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), None, None, True) # perform arithmetic
    for i in range(len(input_f)): # go through all formulas and collect if they can possibly transform
        transformed_equation_list += apply_individual_formula_on_given_equation(tree_form(equation), copy.deepcopy(input_f[i]), copy.deepcopy(output_f[i]))
    return list(set(transformed_equation_list)) # set list to remove duplications

# Function to recursively transform equation
def search(equation, depth):
    if depth == 0: # limit the search
        return None
    output = generate_transformation(equation) # generate equals to the asked one
    for i in range(len(output)):
        result = search(str_form(output[i]), depth-1) # recursively find even more equals
        if result is not None:
            output += result # hoard them
    return output

# Generate all possible equations in mathematics !!!
# Depth is how much complex equation we allow. It can be made as complicated as desired.
def fx_nest(terminal, fx, depth):
    def neighboring_math_equation(curr_tree, depth=depth): # Generate neighbouring equation trees
        def is_terminal(name):
            return not (name in fx.keys()) # Operations are not leaf nodes
        element = None # What to a append to create something new
        def append_at_last(curr_node, depth): # Append something to generate new equation
            if (is_terminal(element) and depth == 0) or (not is_terminal(element) and depth == 1): # The leaf nodes can't be operations
                return None
            if not is_terminal(curr_node.name):
                if len(curr_node.children) < fx[curr_node.name]: # An operation can take only a mentioned number of arugments
                    curr_node.children.append(TreeNode(element))
                    return curr_node
                for i in range(len(curr_node.children)):
                    output = append_at_last(copy.deepcopy(curr_node.children[i]), depth - 1)
                    if output is not None: # Check if the sub tree has already filled with arugments
                        curr_node.children[i] = copy.deepcopy(output)
                        return curr_node
            return None
        new_math_equation_list = []
        for item in terminal + list(fx.keys()): # Create new math equations with given elements
            element = item # set the element we want to use to create new math equation
            tmp = copy.deepcopy(curr_tree)
            result = append_at_last(tmp, depth)
            if result is not None:
                new_math_equation_list.append(result)
        return new_math_equation_list
    all_possibility = []
    # explore mathematics itself with given elements
    # breadth first search, a widely used algorithm
    def bfs(start_node):
        nonlocal all_possibility
        queue = deque()
        visited = set()
        queue.append(start_node)
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                neighbors = neighboring_math_equation(current_node)
                if neighbors == []:
                    all_possibility.append(str_form(current_node))
                    all_possibility = list(set(all_possibility)) # remove duplicates
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
    for item in fx.keys(): # use all the elements
        bfs(TreeNode(item))
    return all_possibility # return mathematical equations produce

# break a equation into parts
def break_equation(equation):
    sub_equation_list = [equation]
    equation = tree_form(equation)
    for child in equation.children: # breaking equation by accessing children
        sub_equation_list += break_equation(str_form(child)) # collect broken equations
    return sub_equation_list

# spot mathematical equations which are poorly formed
def spot_invalid_equation(equation):
    equation = tree_form(equation)
    if equation.name == "f_pow": # power should only have integer on the exponent and it should be two or more than two
        return equation.children[1].name[:2] == "d_" and int(equation.children[1].name[2:]) >= 2
    return True

# fancy print
def print_equation_helper(equation_tree):
    if equation_tree.children == []:
        return equation_tree.name # leaf node
    s = "(" # bracket
    sign = {"f_add": "+", "f_mul": "*", "f_pow": "^"} # operation symbols
    for child in equation_tree.children:
        s+= print_equation_helper(child) + sign[equation_tree.name]
    s = s[:-1] + ")"
    return s

# fancy print main function
def print_equation(eq):
    eq = eq.replace("v_0", "x")
    eq = eq.replace("v_1", "y")
    eq = eq.replace("v_2", "z")
    eq = eq.replace("d_", "")
    return print_equation_helper(tree_form(eq))

# integers start with d and variables start with v
element_list = ["d_" + str(i) for i in range(1, 3)] + ["v_" + str(i) for i in range(0, 1)] # allowed integers and variable in our mathematics

formed_math = fx_nest(element_list, {"f_add": 2, "f_mul": 2, "f_pow": 2}, 2) # scoop out a part of mathematics

formed_math = [equation for equation in formed_math if all(spot_invalid_equation(item) for item in break_equation(equation))] + element_list # remove poorly form math

equal_category = [[item] for item in formed_math] # categories of equal equations

# iterate through all possible equations and categorize equal ones
for equation in formed_math:
    output_list = search(equation, 1) # generate equal ones
    for output in output_list: # check if they are in present in some equality category
        output = str_form(output)
        output_loc = -1
        equation_loc = -1
        for j in range(len(equal_category)):
            if equation in equal_category[j]:
                equation_loc = j
            if output in equal_category[j]:
                output_loc = j
        if equation_loc != -1 and output_loc != -1 and equation_loc != output_loc: # if found two categories with atleast one equation in common
            equal_category.append(equal_category[output_loc]+equal_category[equation_loc]) # merge the two categories
            equal_category.pop(max(output_loc, equation_loc))
            equal_category.pop(min(output_loc, equation_loc))

# print all the equal equation categories
for item in equal_category:
    cat = list(set([print_equation(sub_item) for sub_item in item])) # remove duplicate fancy prints
    for sub_item in cat:
        print(sub_item)
    print("----------")

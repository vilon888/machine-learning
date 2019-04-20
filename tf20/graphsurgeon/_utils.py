import re
# from tensorflow import NodeDef
from tensorflow.core.framework.node_def_pb2 import NodeDef

def _string_matches_regex(match_string, regex):
    # Check for exact matches.
    matches_name = regex == match_string
    if matches_name:
        return True
    # Otherwise, treat as a regex
    matches_regex = False
    re_matches = re.match(regex, match_string)
    # If we find matches...
    if re_matches:
        # Check that they are exact matches
        matches_regex = re_matches.group() == match_string
        if matches_regex:
            return True
    return False

# Takes a string as well as a list of RegExs. Returns true if the string matches any of the RegExs.
def _regex_list_contains_string(regex_list, match_string):
    for regex in regex_list:
        if _string_matches_regex(match_string, regex):
            return True
    return False

# Check if an object is an iterable but NOT a string.
def _is_nonstring_iterable(obj):
    return hasattr(obj, '__contains__') and not type(obj) is str

# Accepts either a string or iterable. If it receives a string, returns a container
# containing that string, otherwise returns the iterable.
def _generate_iterable_for_search(potential_iterable):
    return potential_iterable if _is_nonstring_iterable(potential_iterable) else set([potential_iterable])

# Creates a list of names from a given list of nodes.
def _get_node_names(nodes):
    return [node.name for node in nodes]

def _handle_single_nodes(nodes):
    if isinstance(nodes, NodeDef):
        return [nodes]
    return nodes

# Cleans up the input name. Nodes with multiple outputs are suffixed with a :index_number, so we need to strip this.
def _clean_input_name(input_name):
    return input_name.split(":")[0]

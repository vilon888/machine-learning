try:
    basestring
except NameError:
    basestring = str

def remove_duplicates(graph, names=None):
    '''
    This is a recovery function - if for some reason graph is
    a multigraph (contains nodes of the same name) or contains 
    multiple edges between the same pair of nodes then 
    this function erases duplicates. If `names` is not none then 
    the function erases duplicate edges only between pair of nodes
    that belong to `names`
    
    Args:
        graph - given graph
        names - optional, limits eliminating duplicate edges
        
    Returns:
        Nothing
    '''
    
    if (names is not None):
        names = graph._iterable(names)
    
    duplicated = set()
    to_erase = []
    for node in graph._internal_graphdef.node:
        if str(node.name) in duplicated:
            to_erase.append(node)
        else:
            duplicated.add(str(node.name))
    for node in to_erase:
        graph._internal_graphdef.node.remove(node)
    for node in graph._internal_graphdef.node:
        all_edge = list(node.input)
        distinct_edge = list(set(all_edge))
        for e in all_edge:
            if (names == None):
                node.input.remove(e)
            elif (node.name in names) and (e in names):
                node.input.remove(e)
        for e in distinct_edge:
            if (names == None):
                node.input.append(e)
            elif (node.name in names) and (e in names):
                node.input.append(e)

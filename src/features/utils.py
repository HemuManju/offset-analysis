import collections

from scipy import spatial


def findkeys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
                yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x


def nested_dict():
    return collections.defaultdict(nested_dict)


def _time_kd_tree(time_stamps):
    time_tree = spatial.KDTree(time_stamps)
    return time_tree

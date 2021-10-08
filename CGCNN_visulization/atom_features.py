import numpy as np
from mendeleev import element


NUM_SLICE = 10


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_range(x, allowable_range_tuple, num_slice):
    result = [False for _ in range(num_slice)]
    if not allowable_range_tuple[0] <= x < allowable_range_tuple[1]:
        print("input {} not in allowable range {}".format(x, allowable_range_tuple))
        return result
    result[int((float(x) - allowable_range_tuple[0]) / (allowable_range_tuple[1] - allowable_range_tuple[0]) * num_slice)] =  True
    return result


def element_property(element_name, property_name):
    def log(x):
        try:
            return np.log(x)
        except AttributeError:
            return None
    elem = element(element_name)
    if property_name == 'group':
        if elem.group_id is None:
            return 0
        else:
            return elem.group_id
    elif property_name == 'row':
        return elem.period
    elif property_name == 'electronegativity':
        return elem.electronegativity()
    elif property_name == 'covalent_radius':
        return elem.covalent_radius
    elif property_name == 'nvalence':
        return elem.nvalence()
    elif property_name == 'log_ionenergy':
        return log(elem.ionenergies[1])
    elif property_name == 'electron_affinity':
        return elem.electron_affinity
    elif property_name == 'block':
        return elem.block
    elif property_name == 'log_atomic_volume':
        return log(elem.atomic_volume)


def atom_features(element_name):
    return np.array(
        one_of_k_encoding(element_property(element_name, 'group'), range(0, 19)) +
        one_of_k_encoding(element_property(element_name, 'row'), range(1, 8)) +
        one_of_k_encoding_range(element_property(element_name, 'electronegativity'), (0.5, 4.), NUM_SLICE) +
        one_of_k_encoding_range(element_property(element_name, 'covalent_radius'), (25, 250), NUM_SLICE) +
        one_of_k_encoding(element_property(element_name, 'nvalence'), range(1, 13)) +
        one_of_k_encoding_range(element_property(element_name, 'log_ionenergy'), (1.3, 3.3), NUM_SLICE) +
        one_of_k_encoding_range(element_property(element_name, 'electron_affinity'), (-3, 3.7), NUM_SLICE) +
        one_of_k_encoding(element_property(element_name, 'block'), ['s', 'p', 'd', 'f']) +
        one_of_k_encoding_range(element_property(element_name, 'log_atomic_volume'), (1.5, 4.3), NUM_SLICE))


if __name__ == '__main__':
    print(atom_features('H'))

def flatten(S: list) -> list:
    """ Flatten a nested list into a single (flat) list

    Args:
        S (list): nested list to flatten

    Returns:
        list: a list containing the elements of the nested list without the nesting.
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
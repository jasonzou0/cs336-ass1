def get_pairs(byte_tuple: tuple[bytes]):
    pairs = set() 
    i=0
    while i < len(byte_tuple)-1:
        pair = (byte_tuple[i], byte_tuple[i + 1])
        pairs.add(pair)
        i=i+1
    return pairs

def get_cache_updates(old_bytes: tuple[bytes], new_bytes: tuple[bytes], skip_update: bool = True):
    """
    Process the cache update based on the old and new byte tuples and the merges.
    Returns a tuple of sets: (add_set, update_set, delete_set).
    """
    add_set = set()
    update_set = set()
    delete_set = set()

    # Get pairs from old and new byte tuples
    old_pairs = get_pairs(old_bytes)
    new_pairs = get_pairs(new_bytes)

    # Determine deleted pairs (in old_pairs but not in new_pairs)
    for pair in old_pairs:
        if pair not in new_pairs:
            delete_set.add(pair)

    for pair in new_pairs:
        if pair not in old_pairs:
            # Determine added pairs (in new_pairs but not in old_pairs)
            add_set.add(pair)
        else:
            # Determine updated pairs (in both new and old pairs and old not equal new)
            update_set.add(pair)

    if new_pairs == old_pairs and skip_update:
        update_set = set()  # If no changes, clear update set

    return add_set, update_set, delete_set

# old=(b'a',b'b',b'c',b'd',b'e',b'f',b'a',b'b',b'x',b'c',b'd',b'b',b'c',b'm',b'b',b'c',b'b',b'c')
# new=(b'a',b'bc',b'd',b'e',b'f',b'a',b'b',b'x',b'c',b'd',b'bc',b'm',b'bc',b'bc')
# print(get_pairs(old))
# print(get_pairs(new))
# a,u,d=get_cache_updates(old, new)
# print(f"Add: {a}, \nUpdate: {u}, \nDelete: {d}")
# print("--------------------------------------------------------")


# old= (b' ', b'p', b'h', b'o', b't', b'o', b'g', b'r', b'a', b'p', b'h', b'e', b'r')
# new= (b' ', b'p', b'h', b'o', b't', b'o', b'g', b'r', b'a', b'p', b'he', b'r') 
# print(f"old={old} \nnew={new}")
# print(f"old_pairs={get_pairs(old)}")
# print(f"new_pairs={get_pairs(new)}")
# a,u,d=get_cache_updates(old, new)
# print(f"Add: {a}, \nUpdate: {u}, \nDelete: {d}")
# print("--------------------------------------------------------")


def test_process_cache():
    old=(b'l', b'o', b'w')
    new=(b'l', b'o', b'w')
    print(f"old={old} \nnew={new}")
    print(f"old_pairs={get_pairs(old)}")
    print(f"new_pairs={get_pairs(new)}")

    a,u,d=get_cache_updates(old, new, False)
    print(f"Add: {a}, \nUpdate: {u}, \nDelete: {d}")

    assert a==set()
    assert u=={(b'l', b'o'), (b'o', b'w')}
    assert d==set()
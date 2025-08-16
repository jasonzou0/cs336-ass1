old=('a','b','c','d','e','f','a','b','x','c','d','b','c','m','b','c','b','c')
new=('a','bc','d','e','f','a','b','x','c','d','bc','m','bc','bc')
merges=[1,11,14,16]

# output (add_set, upd_set, del_set)


def get_pairs(byte_tuple: tuple[bytes]):
    pairs = set() 
    i=0
    while i < len(byte_tuple)-1:
        pair = (byte_tuple[i], byte_tuple[i + 1])
        pairs.add(pair)
        i=i+1
    return pairs

def get_cache_updates(old_bytes: tuple[bytes], new_bytes: tuple[bytes]):
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
        # Determine added pairs (in new_pairs but not in old_pairs)
        if pair not in old_pairs:
            add_set.add(pair)
        # Determine updated pairs (in both new and old pairs and old not equal new)
        if pair in old_pairs:
            update_set.add(pair)

    if new_pairs == old_pairs:
        update_set = set()  # If no changes, clear update set

    return add_set, update_set, delete_set

print(get_pairs(old))
print(get_pairs(new))
a,u,d=process_cache_update(old, new)
print(f"Add: {a}, \nUpdate: {u}, \nDelete: {d}")
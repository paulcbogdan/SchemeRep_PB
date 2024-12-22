import random
import pandas as pd
from collections import defaultdict
from functools import cache

def generate_comparison_pairs(num_items, comparisons_per_item, random_seed=0):
    """
    Generate pairs of items for comparison where each item is compared to
    exactly n other unique items, accounting for bidirectional comparisons.

    Args:
        items: Total number of items to be compared
        comparisons_per_item: Number of other items each item should be compared to

    Returns:
        list of tuples: Each tuple contains (item1, item2) representing a comparison
    """
    # Validate inputs
    if comparisons_per_item >= num_items:
        raise ValueError("comparisons_per_item must be less than num_items")
    # Track how many comparisons each item has
    comparison_counts = defaultdict(int)

    # if random_seed is not None:
    random.seed(random_seed)

    # Store all comparison pairs
    comparisons = set()

    # Create list of all possible items
    items = list(range(num_items))

    # Keep trying until all items have enough comparisons
    while min(comparison_counts.get(i, 0) for i in items) < comparisons_per_item:
        # Find items that need more comparisons
        needed_items = [i for i in items if comparison_counts[i] < comparisons_per_item]

        if not needed_items:
            break

        item1 = random.choice(needed_items)

        # Find potential items to compare with
        available_items = [
            i for i in items
            if i != item1
               and comparison_counts[i] < comparisons_per_item
               and (item1, i) not in comparisons
               and (i, item1) not in comparisons
        ]

        if not available_items:
            continue

        item2 = random.choice(available_items)

        # Add the comparison pair
        comparisons.add((min(item1, item2), max(item1, item2)))
        comparison_counts[item1] += 1
        comparison_counts[item2] += 1

    return sorted(list(comparisons))

def get_quick_items_non_sym(items):
    random.seed(0)
    def pop_random(lst):
        idx = random.randrange(0, len(lst))
        return lst.pop(idx)

    lst = list(items)
    if len(lst) % 2 != 0:
        lst += [lst[0]]
    pairs = []
    while len(lst):
        rand1 = pop_random(lst)
        rand2 = pop_random(lst)
        pair = rand1, rand2
        pairs.append(pair)
    return pairs


def generate_comparisons_pairs_d1(items):
    item2item1s = defaultdict(list)
    pairs = get_quick_items_non_sym(items)
    for pair in pairs:
        item2item1s[pair[0]].append(pair[1])
        item2item1s[pair[1]].append(pair[0])
    return item2item1s

@cache
def generate_comparison_pairs_d(items, comparisons_per_item=5):
    if comparisons_per_item == 1:
        return generate_comparisons_pairs_d1(items)
    items = sorted(list(items))
    assert not len(items) % 2 == 1, f'Can\'t handle odd numbers of items: {len(items)=}'
    comparison_pairs = generate_comparison_pairs(len(items), comparisons_per_item)

    item2item1s = {}
    for idx0, idx1 in comparison_pairs:
        item0 = items[idx0]
        item1 = items[idx1]
        if item0 not in item2item1s:
            item2item1s[item0] = []
        if item1 not in item2item1s:
            item2item1s[item1] = []
        item2item1s[item0].append(item1)
        item2item1s[item1].append(item0)
    return item2item1s
    # for item0, l in item2item1s.items():
    #     print(f'{item0}: {l=}')
    #     assert len(l) == comparisons_per_item
    # quit()

# Verify the results
def verify_comparisons(df, num_items, target_comparisons):
    """Verify that each item appears exactly the target number of times"""
    all_items = set(range(num_items))

    # Count appearances of each item
    item_counts = defaultdict(int)
    for _, row in df.iterrows():
        item_counts[row['Item1']] += 1
        item_counts[row['Item2']] += 1

    # Check results
    print(f"\nVerification Results:")
    print(f"Total unique pairs: {len(df)}")
    print(f"Items with incorrect number of comparisons: ")
    for item in all_items:
        count = item_counts.get(item, 0)
        if count != target_comparisons:
            print(f"Item {item}: {count} comparisons")


if __name__ == "__main__":
    num_items = 300
    comparisons_per_item = 1

    comparison_pairs = generate_comparison_pairs_d(tuple(range(30)), comparisons_per_item)
    # for item, l in comparison_pairs.items():
    #     print(f'{item}: {len(l)=}')
    # print(comparison_pairs)
    # quit()
    # print(comparison_pairs[:10])
    # quit()

    # Convert to DataFrame for easy viewing/export
    df = pd.DataFrame(comparison_pairs, columns=['Item1', 'Item2'])
    # Run verification
    verify_comparisons(df, num_items, comparisons_per_item)

    # Example usage:
    print("\nFirst 10 comparison pairs:")
    print(df.head(10))
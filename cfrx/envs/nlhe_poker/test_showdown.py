import jax.numpy as jnp

from cfrx.envs.nlhe_poker.showdown import (
    flush_score,
    four_of_a_kind_score,
    full_house_score,
    get_hand_type,
    high_card_hand_score,
    one_pair_score,
    straight_flush_score,
    straight_score,
    three_of_a_kind_score,
    two_pair_score,
)


def test_straight_flush_order():
    hands = [
        (
            jnp.array([0, 1, 2, 3, 4, 8, 12]),
            jnp.array([0, 0, 0, 0, 0, 1, 2]),
        ),  # 5-high straight flush
        (
            jnp.array([4, 5, 6, 7, 8, 1, 11]),
            jnp.array([2, 2, 2, 2, 2, 3, 0]),
        ),  # 9-high straight flush
        (
            jnp.array([8, 9, 10, 11, 12, 0, 2]),
            jnp.array([1, 1, 1, 1, 1, 2, 3]),
        ),  # King-high straight flush
        (
            jnp.array([9, 10, 11, 12, 0, 8, 7]),
            jnp.array([3, 3, 3, 3, 3, 0, 1]),
        ),  # Ace-high straight flush
    ]

    scores = [straight_flush_score(ranks, suits) for ranks, suits in hands]

    print("straight_flush_score", scores)

    # Expected order of scores from lowest to highest
    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_four_of_a_kind_order():
    hands = [
        jnp.array([10, 10, 10, 10, 0, 1, 2]),  # Four of Jacks with kicker A
        jnp.array([11, 11, 11, 11, 0, 1, 2]),  # Four of Queens with kicker A
        jnp.array([12, 12, 12, 12, 0, 2, 3]),  # Four of Kings with kicker A
        jnp.array([0, 0, 0, 0, 1, 1, 1]),  # Four of Aces with kicker 2
        jnp.array([0, 0, 0, 0, 1, 1, 2]),  # Four of Aces with kicker 3
        jnp.array([0, 0, 0, 0, 1, 1, 12]),  # Four of Aces with kicker King
        jnp.array([0, 0, 0, 0, 11, 12, 12]),  # Four of Aces with kicker King
    ]

    suits = (jnp.array([3, 3, 3, 3, 0, 1, 2]),)
    scores = [four_of_a_kind_score(ranks, suits) for ranks in hands]

    print("four_of_a_kind", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_three_of_a_kind_order():
    hands = [
        (jnp.array([1, 1, 1, 2, 3, 4, 6])),  # Three 2s with kickers (5,7)
        (jnp.array([1, 1, 1, 2, 3, 5, 6])),  # Three 2s with kickers (6,7)
        (jnp.array([1, 1, 1, 2, 3, 5, 0])),  # Three 2s with kickers (6,A)
        (jnp.array([1, 1, 1, 3, 3, 3, 2])),  # Three 2s, three 4s with kickers (2,3)
        (jnp.array([1, 1, 1, 3, 3, 3, 0])),  # Three 2s, three 4s with kickers (2,A)
        (jnp.array([10, 10, 10, 3, 3, 3, 5])),  # Three Js, three 4s with kickers (4,6)
        (jnp.array([10, 10, 10, 3, 3, 3, 5])),  # Three Js, three 4s with kickers (4,A)
        (jnp.array([10, 10, 10, 0, 0, 0, 5])),  # Three Js, three As
    ]

    suits = jnp.array([0, 0, 0, 0, 1, 2, 3])
    scores = [three_of_a_kind_score(ranks, suits) for ranks in hands]

    print("three_of_a_kind", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_flush_order():
    hands = [
        (
            jnp.array([2, 4, 5, 6, 10, 4, 5]),
            jnp.array([0, 0, 0, 0, 0, 1, 1]),
        ),
        (
            jnp.array([2, 4, 5, 8, 10, 4, 5]),
            jnp.array([0, 0, 0, 0, 0, 1, 1]),
        ),
        (
            jnp.array([2, 4, 5, 8, 10, 7, 5]),
            jnp.array([0, 0, 0, 0, 0, 0, 1]),
        ),
        (
            jnp.array([0, 4, 5, 8, 10, 7, 5]),
            jnp.array([0, 0, 0, 0, 0, 0, 1]),
        ),
    ]

    scores = [flush_score(ranks, suits) for ranks, suits in hands]

    print("flush", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_straight_order():
    hands = [
        (jnp.array([0, 1, 2, 3, 4, 7, 8])),
        (jnp.array([0, 1, 2, 3, 4, 7, 0])),
        (jnp.array([3, 4, 5, 6, 7, 10, 11])),
        (jnp.array([2, 9, 8, 9, 10, 11, 12])),
        (jnp.array([2, 9, 9, 10, 11, 12, 0])),
    ]

    suits = jnp.array([0, 0, 0, 0, 1, 2, 3])
    scores = [straight_score(ranks, suits) for ranks in hands]

    print("straight", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_full_house_order():
    hands = [
        (jnp.array([1, 1, 2, 2, 2, 3, 4])),
        (jnp.array([4, 4, 3, 3, 3, 2, 1])),
        (jnp.array([4, 4, 4, 3, 3, 3, 2])),
        (jnp.array([2, 2, 10, 10, 10, 4, 5])),
        (jnp.array([12, 12, 10, 10, 10, 4, 5])),
        (jnp.array([12, 12, 0, 0, 0, 4, 5])),
    ]

    suits = jnp.array([0, 0, 0, 0, 1, 2, 3])
    scores = [full_house_score(ranks, suits) for ranks in hands]

    print("full_house", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_two_pair_order():
    hands = [
        (jnp.array([1, 1, 2, 2, 3, 5, 7])),
        (jnp.array([1, 1, 2, 2, 4, 5, 0])),
        (jnp.array([1, 1, 3, 4, 5, 5, 7])),
        (jnp.array([1, 2, 3, 3, 5, 5, 7])),
        (jnp.array([1, 1, 3, 3, 5, 5, 7])),
        (jnp.array([11, 11, 12, 12, 2, 3, 6])),
        (jnp.array([12, 12, 0, 0, 2, 3, 4])),
        (jnp.array([12, 12, 0, 0, 11, 3, 4])),
    ]
    suits = jnp.array([0, 0, 0, 0, 1, 2, 3])
    scores = [two_pair_score(ranks, suits) for ranks in hands]

    print("two_pair", scores)

    expected_order = sorted(scores)

    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_one_pair_order():
    hands = [
        (jnp.array([1, 1, 2, 3, 4, 7, 8])),
        (jnp.array([1, 1, 2, 3, 9, 7, 8])),
        (jnp.array([1, 1, 4, 5, 0, 7, 8])),
        (jnp.array([12, 12, 11, 10, 9, 1, 2])),
        (jnp.array([0, 0, 2, 3, 4, 5, 7])),
    ]

    suits = jnp.array([0, 0, 0, 0, 1, 2, 3])
    scores = [one_pair_score(ranks, suits) for ranks in hands]

    print("one_pair", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_high_card_order():
    hands = [
        (jnp.array([1, 2, 3, 4, 6, 7, 8])),
        (jnp.array([1, 2, 3, 4, 6, 7, 10])),
        (jnp.array([1, 2, 3, 4, 6, 8, 10])),
        (jnp.array([1, 2, 12, 4, 6, 8, 10])),
        (jnp.array([0, 2, 12, 4, 6, 8, 10])),
    ]
    suits = jnp.array([0, 0, 0, 0, 1, 2, 3])
    scores = [high_card_hand_score(ranks, suits) for ranks in hands]

    print("high_card", scores)

    expected_order = sorted(scores)
    assert scores == expected_order, f"Scores are not in expected order: {scores}"


def test_get_hand_type():
    # Royal Flush (Ace-high straight flush)
    ranks = jnp.asarray([0, 9, 10, 11, 12, 3, 5])  # A, 10, J, Q, K, 4, 6
    suits = jnp.asarray([1, 1, 1, 1, 1, 2, 3])
    assert get_hand_type(ranks, suits) == 8

    # Straight Flush
    ranks = jnp.asarray([8, 9, 10, 11, 12, 3, 5])  # 9, 10, J, Q, K, 4, 6
    suits = jnp.asarray([2, 2, 2, 2, 2, 3, 1])
    assert get_hand_type(ranks, suits) == 8

    # Four of a Kind
    ranks = jnp.asarray([1, 1, 1, 1, 2, 3, 4])  # 2, 2, 2, 2, 3, 4, 5
    suits = jnp.asarray([0, 1, 2, 3, 0, 1, 2])
    assert get_hand_type(ranks, suits) == 7

    # Full House
    ranks = jnp.asarray([2, 2, 2, 3, 3, 4, 5])  # 3, 3, 3, 4, 4, 5, 6
    suits = jnp.asarray([0, 1, 2, 0, 1, 2, 3])
    assert get_hand_type(ranks, suits) == 6

    # Flush
    ranks = jnp.asarray([1, 4, 6, 8, 10, 2, 3])  # 2, 5, 7, 9, J, 3, 4
    suits = jnp.asarray([1, 1, 1, 1, 1, 2, 0])
    assert get_hand_type(ranks, suits) == 5

    # Straight
    ranks = jnp.asarray([0, 9, 10, 11, 12, 1, 2])  # 5, 6, 7, 8, 9, 10, J
    suits = jnp.asarray([0, 1, 2, 3, 0, 1, 2])
    assert get_hand_type(ranks, suits) == 4

    # Three of a Kind
    ranks = jnp.asarray([3, 3, 3, 5, 6, 7, 8])  # 4, 4, 4, 6, 7, 8, 9
    suits = jnp.asarray([0, 1, 2, 0, 1, 2, 3])
    assert get_hand_type(ranks, suits) == 3

    # Two Pairs
    ranks = jnp.asarray([4, 4, 7, 7, 10, 1, 2])  # 5, 5, 8, 8, J, 2, 3
    suits = jnp.asarray([0, 1, 2, 3, 0, 1, 2])
    assert get_hand_type(ranks, suits) == 2

    # One Pair
    ranks = jnp.asarray([6, 6, 8, 9, 11, 1, 3])  # 7, 7, 9, 10, Q, 2, 4
    suits = jnp.asarray([0, 1, 2, 0, 1, 2, 3])
    assert get_hand_type(ranks, suits) == 1

    ranks = jnp.asarray([1, 10, 5, 7, 9, 2, 4])  # 2, 4, 6, 8, 10, 3, 5
    suits = jnp.asarray([0, 1, 2, 3, 0, 1, 2])
    assert get_hand_type(ranks, suits) == 0

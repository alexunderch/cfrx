import jax
import jax.numpy as jnp


def straight_flush_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    suit_counts = jnp.bincount(suits, length=4)
    color = suit_counts.argmax()
    ranks = jnp.where(suits == color, ranks, -1)
    append_ace = jnp.where((ranks == 0).any(), 13, -1)
    values = jnp.unique(jnp.sort(ranks), size=7, fill_value=append_ace)
    diff = jnp.diff(values, append=append_ace)
    conv = jnp.convolve((diff == 1), jnp.ones(4), mode="same")
    straigth_idx = jnp.where(conv == 4, jnp.arange(7), -1).argmax()
    straight_rank = values[straigth_idx]
    return 8_000_000 + straight_rank


def four_of_a_kind_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    rank_counts = jnp.bincount(ranks, length=13)
    active_rank = rank_counts.argmax()
    remaining_ranks = jnp.where(ranks != active_rank, ranks, -1)
    remaining_score = high_card_score(remaining_ranks, suits, n_start=0, n_end=1)
    return 7_000_000 + active_rank * 13 + remaining_score


def full_house_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    rank_counts = jnp.bincount(ranks, length=13)
    three_of_a_kind_mask = rank_counts >= 3
    active_rank_three = jnp.where(three_of_a_kind_mask, jnp.arange(13), -1).argmax()
    rank_counts = rank_counts.at[active_rank_three].set(0)
    pair_mask = rank_counts >= 2
    active_rank_pair = jnp.where(pair_mask, jnp.arange(13), -1).argmax()
    return 6_000_000 + active_rank_three * 13 + active_rank_pair


def flush_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    suit_counts = jnp.bincount(suits, length=4)
    color = suit_counts.argmax()
    colored_ranks = jnp.where(suits == color, ranks, -1)
    colored_ranks = colored_ranks.sort()
    score = high_card_score(colored_ranks, suits)
    return 5_000_000 + score


def straight_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    append_ace = jnp.where((ranks == 0).any(), 13, -1)
    values = jnp.unique(jnp.sort(ranks), size=7, fill_value=append_ace)
    diff = jnp.diff(values, append=append_ace)
    conv = jnp.convolve((diff == 1), jnp.ones(4), mode="same")
    straigth_idx = jnp.where(conv == 4, jnp.arange(7), -1).argmax()
    straight_rank = values[straigth_idx]
    return 4_000_000 + straight_rank


def three_of_a_kind_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    rank_counts = jnp.bincount(ranks, length=13)
    three_of_a_kind_mask = rank_counts >= 3
    active_rank = jnp.where(three_of_a_kind_mask, jnp.arange(13), -1).argmax()
    remaining_ranks = jnp.where(ranks != active_rank, ranks, -1)
    kicker_score = high_card_score(remaining_ranks, suits, n_start=0, n_end=2)
    return 3_000_000 + active_rank * 13**2 + kicker_score


def two_pair_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    rank_counts = jnp.bincount(ranks, length=13)
    pair_mask = rank_counts >= 2
    pair_ranks = jnp.where(pair_mask, jnp.arange(13), -1)
    pair_ranks = jnp.argsort(pair_ranks)[-2:]
    pair_ranks = jnp.sort(pair_ranks)[::-1]
    active_rank_first_pair, active_rank_second_pair = (
        pair_ranks[0],
        pair_ranks[1],
    )
    rank_counts = rank_counts.at[active_rank_first_pair].set(0)
    rank_counts = rank_counts.at[active_rank_second_pair].set(0)
    remaining_ranks = jnp.where(
        (ranks != active_rank_first_pair) & (ranks != active_rank_second_pair),
        ranks,
        -1,
    )
    kicker_score = high_card_score(remaining_ranks, suits, n_start=0, n_end=1)
    return (
        2_000_000
        + active_rank_first_pair * 13**2
        + active_rank_second_pair * 13
        + kicker_score
    )


def one_pair_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    rank_counts = jnp.bincount(ranks, length=13)
    pair_mask = rank_counts >= 2
    active_rank_pair = jnp.where(pair_mask, jnp.arange(13), -1).argmax()
    rank_counts = rank_counts.at[active_rank_pair].set(0)
    remaining_ranks = jnp.where(ranks != active_rank_pair, ranks, -1)
    kicker_score = high_card_score(remaining_ranks, suits, n_start=0, n_end=3)
    return 1_000_000 + active_rank_pair * 13**3 + kicker_score


def high_card_hand_score(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    ranks = jnp.where(ranks == 0, 12, ranks - 1)
    return high_card_score(ranks, suits, n_start=0, n_end=5)


def high_card_score(
    ranks: jax.Array, suits: jax.Array, n_start: int = 0, n_end: int = 5
) -> jax.Array:
    n = n_end - n_start
    ranks = ranks.sort()
    rank_scores = jnp.array([13**k for k in range(n_start, n_end)])
    ranks = ranks[-n:]
    return (ranks * rank_scores).sum()


def is_straight_fn(ranks):
    append_ace = jnp.where((ranks == 0).any(), 13, -1)
    values = jnp.unique(jnp.sort(ranks), size=7, fill_value=append_ace)
    diff = jnp.diff(values, append=append_ace)
    cond = (jnp.convolve((diff == 1), jnp.ones(4), mode="valid") == 4).any()
    return cond


def is_straight_flush_fn(ranks, suits):
    suit_counts = jnp.bincount(suits, length=4)
    color = suit_counts.argmax()
    ranks = jnp.where(suits == color, ranks, -1)
    return is_straight_fn(ranks)


def get_hand_type(ranks: jax.Array, suits: jax.Array) -> jax.Array:
    rank_counts = jnp.bincount(ranks, length=13)
    suit_counts = jnp.bincount(suits, length=4)

    is_straight_flush = is_straight_flush_fn(ranks, suits)
    higher = is_straight_flush
    index = 8 * is_straight_flush

    is_four_of_a_kind = ~higher & (rank_counts == 4).any()
    higher |= is_four_of_a_kind
    index += 7 * is_four_of_a_kind

    is_full = ~higher & (rank_counts == 3).any() & (rank_counts == 2).any()
    higher |= is_full
    index += 6 * is_full

    is_flush = ~higher & (suit_counts >= 5).any()
    higher |= is_flush
    index += 5 * is_flush

    is_straight = ~higher & is_straight_fn(ranks)
    higher |= is_straight
    index += 4 * is_straight

    is_three_of_a_kind = ~higher & (rank_counts == 3).any()
    higher |= is_three_of_a_kind
    index += 3 * is_three_of_a_kind

    is_two_pairs = ~higher & ((rank_counts == 2).sum() >= 2)
    higher |= is_two_pairs
    index += 2 * is_two_pairs

    is_one_pair = ~higher & (rank_counts == 2).any()
    higher |= is_one_pair
    index += 1 * is_one_pair

    return index


def get_showdown_score(hand: jax.Array) -> jax.Array:
    hand = hand.astype(jnp.int32)
    ranks = hand % 13
    suits = hand // 13

    hand_type = get_hand_type(ranks, suits)

    score = jax.lax.switch(
        hand_type,
        [
            high_card_hand_score,
            one_pair_score,
            two_pair_score,
            three_of_a_kind_score,
            straight_score,
            flush_score,
            full_house_score,
            four_of_a_kind_score,
            straight_flush_score,
        ],
        ranks,
        suits,
    )

    return score

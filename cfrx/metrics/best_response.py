from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from cfrx.tree import Tree


def backward_one_infoset(
    tree: Tree,
    info_states: Array,
    current_infoset: Int[Array, ""],
    br_player: int,
    depth: Int[Array, ""],
) -> Tree:
    # Select all nodes in this infoset
    infoset_mask = (
        (tree.depth == depth)
        & (~tree.states.terminated)
        & (info_states == current_infoset)
    )

    is_br_player = (
        (tree.states.current_player == br_player)
        & (~tree.states.chance_node)
        & infoset_mask
    ).any()

    p_opponent = tree.extra_data["p_opponent"]
    p_chance = tree.extra_data["p_chance"]
    p_self = tree.extra_data["p_self"]

    # Get expected values for each of the node in the infoset
    cf_reach_prob = (p_opponent * p_chance * p_self)[..., None]

    legal_action_mask = (tree.states.legal_action_mask & infoset_mask[..., None]).any(
        axis=0
    )

    best_response_values = tree.children_values[..., br_player] * cf_reach_prob  # (T, a)
    best_response_values = jnp.sum(
        best_response_values, axis=0, where=infoset_mask[..., None]  # (a,)
    )

    best_action = jnp.where(legal_action_mask, best_response_values, -jnp.inf).argmax()

    br_value = tree.children_values[:, best_action, br_player]

    expected_current_value = (
        tree.children_values[..., br_player] * tree.children_prior_logits
    ).sum(axis=1)

    current_value = jnp.where(is_br_player, br_value, expected_current_value)

    new_node_values = jnp.where(
        infoset_mask, current_value, tree.node_values[..., br_player]
    )
    new_children_values = jnp.where(
        tree.children_index != -1, new_node_values[tree.children_index], 0
    )

    tree = tree._replace(
        node_values=tree.node_values.at[..., br_player].set(new_node_values),
        children_values=tree.children_values.at[..., br_player].set(new_children_values),
    )

    return tree


def backward_one_depth_level(
    tree: Tree,
    depth: Int[Array, ""],
    br_player: int,
    info_state_fn: Callable,
) -> Tree:
    info_states = jax.vmap(info_state_fn)(tree.states.info_state)

    def cond_fn(val: tuple[Tree, Array]) -> Array:
        tree, visited = val
        nodes_to_visit_idx = (
            (tree.depth == depth) & (~tree.states.terminated) & (~visited)
        )
        return nodes_to_visit_idx.any()

    def loop_fn(val: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, visited = val
        nodes_to_visit_idx = (
            (tree.depth == depth) & (~tree.states.terminated) & (~visited)
        )

        # Select an infoset to resolve
        selected_infoset_idx = nodes_to_visit_idx.argmax()
        selected_infoset = info_states[selected_infoset_idx]

        tree = backward_one_infoset(
            tree=tree,
            depth=depth,
            info_states=info_states,
            current_infoset=selected_infoset,
            br_player=br_player,
        )

        visited = jnp.where(info_states == selected_infoset, True, visited)
        return tree, visited

    visited = jnp.zeros(tree.node_values.shape[0], dtype=bool)
    tree, visited = jax.lax.while_loop(cond_fn, loop_fn, (tree, visited))

    return tree


def compute_best_response_value(
    tree: Tree,
    br_player: int,
    info_state_fn: Callable,
) -> Float[Array, " num_players"]:
    depth = tree.depth.max()

    def cond_fn(val: tuple[Tree, Array]) -> Array:
        _, depth = val
        return depth >= 0

    def loop_fn(val: tuple[Tree, Array]) -> tuple[Tree, Array]:
        tree, depth = val
        tree = backward_one_depth_level(
            tree=tree, depth=depth, br_player=br_player, info_state_fn=info_state_fn
        )
        depth -= 1
        return tree, depth

    tree, _ = jax.lax.while_loop(cond_fn, loop_fn, (tree, depth))
    return tree.node_values[0]

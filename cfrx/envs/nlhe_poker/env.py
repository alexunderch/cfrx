from typing import NamedTuple

import jax
import jax.numpy as jnp

import cfrx
from cfrx.envs.nlhe_poker.showdown import get_showdown_score

NUM_PLAYERS = 2


class State(NamedTuple):
    board: jax.Array
    board_mask: jax.Array
    hands: jax.Array
    stacks: jax.Array
    bets: jax.Array
    bets_mask: jax.Array
    current_player: jax.Array
    min_bet: jax.Array
    min_raise: jax.Array
    current_round: jax.Array
    terminated: jax.Array
    small_blind: jax.Array
    showdown_result: jax.Array
    dealer: jax.Array
    rewards: jax.Array


class InfoState(NamedTuple):
    hand: jax.Array
    board: jax.Array
    board_mask: jax.Array
    bets: jax.Array
    bets_mask: jax.Array
    stacks: jax.Array


class TexasHoldem:
    def __init__(self, num_max_bets: int = 4):
        self.num_max_bets = num_max_bets
        self._num_visibles = jnp.array([0, 3, 4, 5], dtype=jnp.uint8)

    @classmethod
    def action_to_string(cls, action: jax.Array) -> str:
        raise NotImplementedError

    @property
    def max_episode_length(self) -> int:
        return 4 * self.num_max_bets

    @property
    def max_nodes(self) -> int:
        raise NotImplementedError

    @property
    def n_info_states(self) -> int:
        raise NotImplementedError

    def init(
        self,
        random_key: jax.Array,
        dealer_player: int = 0,
        initial_stacks: jax.Array | None = None,
        small_blind: float = 1.0,
    ):
        deck = jax.random.permutation(random_key, jnp.arange(52, dtype=jnp.uint8))

        hands = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.uint8)
        for i in range(NUM_PLAYERS):
            cards = deck[i * 2 : (i + 1) * 2]
            hands = hands.at[i].set(cards)

        board = deck[NUM_PLAYERS * 2 : NUM_PLAYERS * 2 + 5].astype(jnp.uint8)

        board_mask = jnp.array(0)

        if initial_stacks is None:
            stacks = jnp.ones(NUM_PLAYERS, dtype=float) * 100 * small_blind
        else:
            stacks = initial_stacks

        sb_player = (dealer_player + 1) % NUM_PLAYERS
        bb_player = (dealer_player + 2) % NUM_PLAYERS
        first_player = (dealer_player + 3) % NUM_PLAYERS

        bets = jnp.zeros((4, NUM_PLAYERS, self.num_max_bets), dtype=float)

        sb_bet = jnp.minimum(small_blind, stacks[sb_player])
        bb_bet = jnp.minimum(small_blind * 2, stacks[bb_player])
        bets = bets.at[0, sb_player, 0].set(sb_bet)
        bets = bets.at[0, bb_player, 0].set(bb_bet)
        stacks = stacks.at[sb_player].set(stacks[sb_player] - sb_bet)
        stacks = stacks.at[bb_player].set(stacks[bb_player] - bb_bet)

        bets_mask = bets > 0.0

        showdown_result = self._resolve_showdown(board, hands)

        state = State(
            board=board,
            board_mask=board_mask,
            hands=hands,
            stacks=stacks,
            bets=bets,
            bets_mask=bets_mask,
            current_player=jnp.array(first_player),
            min_bet=jnp.array(small_blind),
            min_raise=2 * jnp.array(small_blind),
            current_round=jnp.array(0),
            terminated=jnp.array(False),
            small_blind=jnp.array(small_blind),
            showdown_result=showdown_result,
            dealer=jnp.array(dealer_player),
            rewards=jnp.array([0.0, 0.0]),
        )

        return state

    def _is_end_round(
        self,
        current_round: jax.Array,
        bets_mask: jax.Array,
        bets: jax.Array,
    ) -> jax.Array:
        bets_mask = jnp.where(
            current_round == 0, bets_mask.at[0, :, 0].set(False), bets_mask
        )
        has_everyone_spoken = bets_mask[current_round].any(axis=1).all()
        total_bets = bets[current_round].sum(axis=-1)
        same_bets = (total_bets.max() == total_bets.min()).all()

        end_round = has_everyone_spoken & same_bets
        return end_round

    def _resolve_showdown(self, board: jax.Array, hands: jax.Array) -> jax.Array:
        """
        Return 0 if p0 wins, 1 if p1 wins, -1 if tie
        """

        p0_score = get_showdown_score(jnp.concatenate([board, hands[0]]))
        p1_score = get_showdown_score(jnp.concatenate([board, hands[1]]))

        return jnp.where(p0_score == p1_score, jnp.array(-1), p0_score < p1_score)

    def _current_round_to_board_mask(self, current_round: jax.Array) -> jax.Array:
        return self._num_visibles[current_round]

    def step(self, state: State, action: jax.Array) -> State:
        fold = action < 0.0

        # check = action == 0.0
        # bet = action > 0.0

        # clit bet
        action = jnp.clip(action, a_min=0, a_max=state.stacks[state.current_player])

        cp = state.current_player
        cr = state.current_round

        # store bet, update bet mask, update stack

        current_idx = state.bets_mask[cr, cp].sum()
        new_bets_mask = state.bets_mask.at[cr, cp, current_idx].set(True)
        new_bets = state.bets.at[cr, cp, current_idx].set(action)
        new_stacks = state.stacks.at[cp].set(state.stacks[cp] - action)

        reward = jnp.ones(NUM_PLAYERS) * new_bets[:, cp].sum()
        # compute hypothetic fold reward
        fold_reward = jnp.where(jnp.arange(NUM_PLAYERS) == cp, -reward, reward)
        # jax.debug.print("fold_reward: {x}", x=fold_reward)

        # compute hypothetic showdown reward
        showdown_result = state.showdown_result
        showdown_reward = jnp.where(
            jnp.arange(NUM_PLAYERS) == showdown_result, reward, -reward
        )
        showdown_reward = jnp.where(showdown_result == -1, 0, showdown_reward)

        end_round = self._is_end_round(
            current_round=state.current_round,
            bets_mask=new_bets_mask,
            bets=new_bets,
        )

        # jax.debug.print("end_round: {x}", x=end_round)

        is_allin = (state.stacks == 0).any() & ~fold

        is_showdown = (end_round & (state.current_round == 3)) | is_allin

        done = state.terminated | is_showdown | fold

        # jax.debug.print("is showdown: {x}", x=is_showdown)

        reward = jnp.where(is_showdown, showdown_reward, fold_reward)
        reward = jnp.where(done, reward, 0.0)

        new_cr = jnp.where(end_round, cr + 1, cr)
        new_cr = jnp.where(is_showdown, 4, new_cr)

        new_current_player = jnp.where(
            end_round, (state.dealer + 1) % NUM_PLAYERS, (cp + 1) % NUM_PLAYERS
        )

        bet_diff = jnp.abs(jnp.diff(new_bets[cr].sum(axis=-1)))[0]
        my_total_bet = new_bets[cr, cp].sum(axis=-1)

        min_bet = jnp.maximum(bet_diff, state.small_blind)
        min_bet = jnp.where(end_round, state.small_blind, min_bet)

        min_raise = jnp.where(end_round, min_bet, my_total_bet + min_bet)

        new_state = State(
            board=state.board,
            board_mask=self._num_visibles[new_cr],
            hands=state.hands,
            stacks=new_stacks,
            bets=new_bets,
            bets_mask=new_bets_mask,
            current_player=new_current_player,
            min_bet=min_bet,
            min_raise=min_raise,
            current_round=new_cr,
            terminated=done,
            small_blind=state.small_blind,
            showdown_result=state.showdown_result,
            dealer=state.dealer,
            rewards=reward,
        )

        return new_state

    def observe(self, state: State) -> InfoState:
        board_mask = self._num_visibles[state.current_round]
        board_mask = ~(
            jnp.zeros(5, dtype=bool).at[board_mask].set(True).cumsum().astype(bool)
        )
        infostate = InfoState(
            hand=state.hands[state.current_player],
            board=jnp.where(board_mask, state.board, -1),
            board_mask=board_mask,
            bets=state.bets,
            bets_mask=state.bets_mask,
            stacks=state.stacks,
        )

        return infostate

import jax
from jaxtyping import Array, Int
from tqdm import tqdm

from cfrx.algorithms.cfr.cfr import CFRState, do_iteration
from cfrx.envs import Env
from cfrx.metrics import exploitability
from cfrx.policy import TabularPolicy


class CFRTrainer:
    def __init__(self, env: Env, policy: TabularPolicy, device: str):
        self._env = env
        self._policy = policy
        self._device = jax.devices(device)[0]
        self._exploitability_fn = jax.jit(
            lambda policy_params: exploitability(
                policy_params=policy_params,
                env=env,
                n_players=env.n_players,
                n_max_nodes=env.max_nodes,
                policy=policy,
            ),
            device=self._device,
        )

        self._do_iteration_fn = jax.jit(
            lambda training_state, update_player: do_iteration(
                training_state=training_state,
                env=env,
                policy=policy,
                update_player=update_player,
            ),
            device=self._device,
        )

    def do_n_iterations(
        self,
        training_state: CFRState,
        update_player: Int[Array, ""],
        n: int,
    ) -> tuple[CFRState, Int[Array, ""]]:
        def _scan_fn(carry, unused):
            training_state, update_player = carry

            update_player = (update_player + 1) % 2
            training_state = self._do_iteration_fn(
                training_state,
                update_player=update_player,
            )

            return (training_state, update_player), None

        (new_training_state, last_update_player), _ = jax.lax.scan(
            _scan_fn,
            (training_state, update_player),
            None,
            length=n,
        )

        return new_training_state, last_update_player

    def train(
        self, n_iterations: int, metrics_period: int
    ) -> tuple[CFRState, dict[str, Array]]:
        training_state = CFRState.init(self._env.n_info_states, self._env.n_actions)

        assert n_iterations % metrics_period == 0

        n_loops = n_iterations // metrics_period
        update_player = 0
        _do_n_iterations = jax.jit(
            lambda training_state, update_player: self.do_n_iterations(
                training_state=training_state,
                update_player=update_player,
                n=2 * metrics_period,
            ),
            device=self._device,
        )
        metrics = []

        pbar = tqdm(total=n_iterations, desc="Training", unit_scale=True)
        for k in range(n_loops):
            if k == 0:
                current_policy = training_state.avg_probs
                current_policy /= training_state.avg_probs.sum(axis=-1, keepdims=True)
                exp = self._exploitability_fn(policy_params=current_policy)
                metrics.append({"exploitability": exp, "step": 0})
                pbar.set_postfix(exploitability=f"{exp:.1e}")

            # Do n iterations
            training_state, update_player = _do_n_iterations(
                training_state, update_player
            )

            # Evaluate exploitability
            current_policy = training_state.avg_probs
            current_policy /= training_state.avg_probs.sum(axis=-1, keepdims=True)

            exp = self._exploitability_fn(policy_params=current_policy)
            metrics.append({"exploitability": exp, "step": k * metrics_period})
            pbar.set_postfix(exploitability=f"{exp:.1e}")
            pbar.update(metrics_period)

        metrics = jax.tree_map(lambda *x: jax.numpy.stack(x), *metrics)
        return training_state, metrics

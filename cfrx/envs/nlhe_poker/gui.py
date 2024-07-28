from typing import Any

import jax
import jax.numpy as jnp
from nicegui import app, ui

from cfrx.envs.nlhe_poker.env import State, TexasHoldem

SCALE = 0.5
X0 = 400
X1 = 80
Y1 = 700
OFF = 180
DEALER_COORDS = [(3 * X0, 0), (3 * X0, 2500)]
STACK_COORDS = [(120, 50), (120, 400)]
BET_COORDS = [(220, 140), (220, 320)]
POT_COORDS = (500, 220)

COLORS = ["heart", "diamond", "club", "spade"]
VALUES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"]


def create_svg(
    hands: list[tuple[int, int]], board: list[int], board_mask: int, dealer: int
) -> str:
    svg_open = """<svg
      style="position: absolute; top: 0; left: 0; background-color: #f0f0f0"
      width="560"
      height="560"
      xmlns="http://www.w3.org/2000/svg"
      xmlns:xlink="http://www.w3.org/1999/xlink">"""

    svg_close = "</svg>"

    svg_hands = ""
    for i, hand in enumerate(hands):
        for j, card in enumerate(hand):
            x = X0 + j * 40
            y = i * Y1
            svg_hands += f"""
            <use href="static/svg-cards.svg#{COLORS[card//13]}_{VALUES[card%13]}"
            x="{x}" y="{y}" transform="scale({SCALE:.2f})"/>
            """

    svg_board = ""
    for i, card in enumerate(board):
        x = X1 + i * OFF
        y = 350
        if i >= board_mask:
            svg_board += f"""
            <use href="static/svg-cards.svg#alternate-back"
            x="{x}" y="{y}" transform="scale({SCALE:.2f})"/>
            """
        else:
            svg_board += f"""
            <use href="static/svg-cards.svg#{COLORS[card//13]}_{VALUES[card%13]}"
            x="{x}" y="{y}" transform="scale({SCALE:.2f})"/>
            """

    dealer_coords = DEALER_COORDS[dealer]
    svg_dealer = f"""
    <use href="static/svg-cards.svg#joker_black" x="{dealer_coords[0]}"
    y="{dealer_coords[1]}" transform="scale(0.14)"/>
    """

    return f"{svg_open}{svg_hands}{svg_dealer}{svg_board}{svg_close}"


def create_stack(player: int, stack: int) -> str:
    stack_html = f"""<p style='position: absolute; top: {STACK_COORDS[player][1]}px;
    left: {STACK_COORDS[player][0]}px; text-align:center;'>
    Player {player} <br> Stack: {stack}</p>"""

    return stack_html


def create_pot(pot: int) -> str:
    pot_html = f"""<p style='position: absolute; top: {POT_COORDS[1]}px;
    left: {POT_COORDS[0]}px; text-align:center;'>
    Pot: {pot}</p>"""

    return pot_html


def create_bet(player: int, bet: str, current_player: int) -> str:
    color = "red" if player == current_player else None

    bet_html = f"""<div style='position: absolute; top: {BET_COORDS[player][1]}px;
    left: {BET_COORDS[player][0]}px; text-align:center; border-style: solid;
    border-width: 2px; border-color: {color}; width: 60px; height: 20px;
    border-radius: 5px; background-color: #f0f0f0;
    '>
    {bet}</div>"""

    return bet_html


def create_terminated(done: bool, reward: tuple[float, float]) -> str:
    if done:
        end_html = f"""<div style='position: absolute; top: 100px;
        left: 450px; text-align:center;'>Terminated <br> P0: {reward[0]}
        <br> P1: {reward[1]}
        </div>"""
    else:
        end_html = ""

    return end_html


def create_min_bet(min_bet: int, min_raise: int) -> str:
    min_bet_html = f"""<div style='position: absolute; top: 400px;
    left: 450px; text-align:center; width: 80px; height: 40px'>Min bet: <br>{min_bet}
    <br> Min raise: <br>{min_raise }
    </div>"""

    return min_bet_html


def visualize(state: State, container: Any) -> None:
    with container:
        hands = [tuple(state.hands[0]), tuple(state.hands[1])]
        board = state.board.tolist()

        svg = create_svg(
            hands=hands,
            board=board,
            board_mask=int(state.board_mask),
            dealer=int(state.dealer),
        )
        ui.html(svg)

        stack_0 = int(state.stacks[0])

        stack = create_stack(0, stack_0)
        ui.html(stack)

        stack_1 = int(state.stacks[1])

        stack = create_stack(1, stack_1)
        ui.html(stack)

        pot_value = int(state.bets.sum())

        pot = create_pot(pot_value)
        ui.html(pot)

        current_bets = state.bets[state.current_round]
        current_mask = state.bets_mask[state.current_round]

        idx_0 = current_mask[0].sum()

        if idx_0 == 0:
            current_bet_0 = ""
        else:
            current_bet_0 = str(current_bets[0, :idx_0].sum())

        idx_1 = current_mask[1].sum()
        if idx_1 == 0:
            current_bet_1 = ""
        else:
            current_bet_1 = str(current_bets[1, :idx_1].sum())

        cr = int(state.current_player)

        bet = create_bet(0, current_bet_0, current_player=cr)
        ui.html(bet)

        bet = create_bet(1, current_bet_1, current_player=cr)
        ui.html(bet)

        done = create_terminated(bool(state.terminated), reward=tuple(state.rewards))
        ui.html(done)

        min_bet = create_min_bet(int(state.min_bet), int(state.min_raise))
        ui.html(min_bet)


def visualize_previous():
    global states
    global current_state_idx
    global container

    current_state_idx = max(current_state_idx - 1, 0)
    visualize(states[current_state_idx], container=container)


def visualize_next():
    global states
    global current_state_idx
    global container

    current_state_idx = min(current_state_idx + 1, len(states) - 1)
    visualize(states[current_state_idx], container=container)


def delete_last():
    global states
    global current_state_idx
    global container
    if len(states) > 1:
        states.pop(-1)
        current_state_idx = max(len(states) - 1, 0)
        visualize(states[current_state_idx], container=container)


def ui_bet():
    global states
    global current_state_idx
    global env
    global slider
    global container

    state = env.step(states[-1], action=jnp.array(slider.value))
    states.append(state)
    current_state_idx = len(states) - 1
    visualize(states[current_state_idx], container=container)


if __name__ in {"__main__", "__mp_main__"}:
    app.add_static_files("/static", "static")
    env = TexasHoldem()

    container = ui.card().tight()

    with ui.row().style("position: absolute; top:600px"):
        with ui.button_group():
            ui.button("<", on_click=visualize_previous)
            ui.button(">", on_click=visualize_next)
        ui.button("DEL", on_click=delete_last)

        slider = ui.slider(min=-1, max=100, value=0).style("width: 200px")
        ui.label().bind_text_from(slider, "value")
        ui.button("BET", on_click=ui_bet)

    state = env.init(jax.random.PRNGKey(1), initial_stacks=jnp.array([100, 50]))

    states = [state]

    state = env.step(state, action=jnp.array(1.0))
    states.append(state)

    state = env.step(state, action=jnp.array(10.0))
    states.append(state)
    visualize(state, container=container)
    current_state_idx = len(states) - 1

    ui.run(show=False)

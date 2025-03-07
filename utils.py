import random
from typing import Callable, Dict, List

from constants import Action, State


def dict_get(d: Dict[str, float]) -> Callable[[str], float]:
    return lambda k: d[k]


def get_best_action(
    state: State,
    Q_table: Dict[State, Dict[str, float]],
    actions: List[str],
    random_state: random.Random,
) -> Action:
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in actions}
        return random_state.choice(actions)

    max_value = max(Q_table[state].values())
    if all(v == max_value for v in Q_table[state].values()):
        return random_state.choice(list(Q_table[state].keys()))

    return max(Q_table[state], key=dict_get(Q_table[state]))

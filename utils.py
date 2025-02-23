import random
from dataclasses import dataclass
from typing import Callable, Dict, List

from snake_env import Action, State


@dataclass(frozen=True)
class SnakeActions:
    UP: str = "UP"
    DOWN: str = "DOWN"
    LEFT: str = "LEFT"
    RIGHT: str = "RIGHT"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]


def dict_get(d: Dict[str, float]) -> Callable[[str], float]:
    return lambda k: d[k]


def get_best_action(
    state: State,
    Q_table: Dict[State, Dict[str, float]],
    actions: List[str],
    allow_random_on_tie: bool = False,
) -> Action:
    if state not in Q_table:
        Q_table[state] = {a: 0 for a in actions}
        return random.choice(actions)

    if allow_random_on_tie:
        max_value = max(Q_table[state].values())
        if all(v == max_value for v in Q_table[state].values()):
            return random.choice(actions)

    return max(Q_table[state], key=dict_get(Q_table[state]))

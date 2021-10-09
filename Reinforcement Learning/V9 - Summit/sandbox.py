from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    SHORT = 1
    LONG = 2


actions = {
    Action.HOLD: lambda: print("Nolo"),
    Action.LONG: lambda: print("Colo"),
}


actions[Action.HOLD]()
actions[Action.LONG]()

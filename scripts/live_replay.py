"""Replay teacher skills in a loop for live visual observation.

Usage (with scripts/live_viewer.py running in another terminal):

    uv run python scripts/live_replay.py --seed 7 --skill pick --object mug --arm B
    uv run python scripts/live_replay.py --seed 0 --skill pick --object fork_1 --arm A --drawer-open
    uv run python scripts/live_replay.py --seed 2 --skill open_drawer --arm A

Each iteration rebuilds the scene deterministically, runs the skill once,
prints the outcome, pauses briefly so the failure/success frame can be
inspected in the viewer, then repeats.
"""

from __future__ import annotations

import argparse
import sys
import time

from dinner_table.scene.builder import Scene
from dinner_table.teacher.skills import (
    CloseDrawer,
    Home,
    OpenDrawer,
    Pick,
    Place,
    run_skill,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", default="dr_train")
    parser.add_argument("--skill", default="pick",
                        choices=["pick", "place", "open_drawer", "close_drawer", "home"])
    parser.add_argument("--object", default="mug")
    parser.add_argument("--arm", default="B", choices=["A", "B"])
    parser.add_argument("--target", default="placemat_1")
    parser.add_argument("--drawer-open", action="store_true",
                        help="open the drawer via its actuator before the skill")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--pause", type=float, default=4.0,
                        help="seconds to hold the final frame before repeating")
    args = parser.parse_args()

    for i in range(args.iters):
        scene = Scene(seed=args.seed, dr_profile=args.profile)
        if args.drawer_open:
            import mujoco

            act = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drawer_actuator")
            scene.data.ctrl[act] = 0.12
            scene.settle(3.0)
        if args.skill == "pick":
            skill = Pick(args.arm, args.object)
        elif args.skill == "place":
            skill = Place(args.arm, args.object, args.target)
        elif args.skill == "open_drawer":
            skill = OpenDrawer(args.arm)
        elif args.skill == "close_drawer":
            skill = CloseDrawer(args.arm)
        else:
            skill = Home(args.arm)
        result = run_skill(scene, skill)
        print(f"[iter {i + 1}/{args.iters}] seed {args.seed} {args.skill} "
              f"{args.object} arm {args.arm}: {result}", flush=True)
        time.sleep(args.pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())

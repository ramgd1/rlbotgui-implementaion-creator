import os
import shutil
import re
import sys

# ── Enable ANSI colors on Windows ─────────────────────────────────────────────
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )

# ── Colors ─────────────────────────────────────────────────────────────────────
R   = "\033[0m"
CY  = "\033[31m"     # dark red  (main accent)
GR  = "\033[31m"     # dark red  (success)
YL  = "\033[90m"     # dark gray (hints/next step)
MG  = "\033[31m"     # dark red  (credit)
RD  = "\033[31m"     # dark red  (errors)
DM  = "\033[90m"     # dark gray (dim text)
BD  = "\033[1m"      # bold
BL  = "\033[31m"     # dark red  (borders)
WHT = "\033[37m"     # light gray (input text)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

STATIC_FILES = [
    "util",
    "discrete_policy.py",
    "your_act.py",
    "appearance.cfg",
    "requirements.txt",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
def cls():
    os.system("cls" if sys.platform == "win32" else "clear")

def banner():
    print(f"{CY}{BD}")
    print(r"  ▄▄▄▄    ▒█████  ▄▄▄█████▓     ██████ ▓█████▄▄▄█████▓ █    ██  ██▓███  ")
    print(r"  ▓█████▄ ▒██▒  ██▒▓  ██▒ ▓▒   ▒██    ▒ ▓█   ▀▓  ██▒ ▓▒ ██  ▓██▒▓██░  ██▒")
    print(r"  ▒██▒ ▄██▒██░  ██▒▒ ▓██░ ▒░   ░ ▓██▄   ▒███  ▒ ▓██░ ▒░▓██  ▒██░▓██░ ██▓▒")
    print(r"  ▒██░█▀  ▒██   ██░░ ▓██▓ ░      ▒   ██▒▒▓█  ▄░ ▓██▓ ░ ▓▓█  ░██░▒██▄█▓▒ ▒")
    print(r"  ░▓█  ▀█▓░ ████▓▒░  ▒██▒ ░    ▒██████▒▒░▒████▒ ▒██▒ ░ ▒▒█████▓ ▒██▒ ░  ░")
    print(r"  ░▒▓███▀▒░ ▒░▒░▒░   ▒ ░░      ▒ ▒▓▒ ▒ ░░░ ▒░ ░ ▒ ░░   ░▒▓▒ ▒ ▒ ▒▓▒░ ░  ░")
    print(r"  ▒░▒   ░   ░ ▒ ▒░     ░       ░ ░▒  ░ ░ ░ ░  ░   ░    ░░▒░ ░ ░ ░▒ ░     ")
    print(r"   ░    ░ ░ ░ ░ ▒    ░         ░  ░  ░     ░    ░       ░░░ ░ ░ ░░       ")
    print(r"   ░          ░ ░                    ░     ░  ░           ░              ")
    print(r"         ░                                                                 ")
    print(f"{R}")
    print(f"  {DM}{'─' * 51}{R}")
    print(f"  {DM}rlbot bot generator{R}  {MG}by q7cj(ram){R}")
    print(f"  {DM}{'─' * 51}{R}")
    print()

def section(label):
    print(f"\n  {BL}╔{'═' * (len(label) + 2)}╗{R}")
    print(f"  {BL}║{R} {BD}{WHT}{label}{R} {BL}║{R}")
    print(f"  {BL}╚{'═' * (len(label) + 2)}╝{R}")

def ask(question, validator=None, hint=""):
    hint_str = f"  {DM}{hint}{R}" if hint else ""
    while True:
        answer = input(f"  {CY}>{R}  {WHT}{question}{R}{hint_str}  ").strip()
        if not answer:
            print(f"  {RD}  empty -- try again{R}")
            continue
        if validator is None or validator(answer):
            return answer
        print(f"  {RD}  invalid -- try again{R}")

def ask_layers(label):
    while True:
        raw = ask(label, hint="· 1024,1024,512")
        try:
            layers = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if layers:
                return layers
        except ValueError:
            pass
        print(f"  {RD}  use comma-separated integers{R}")

def tag(text, color=CY):
    return f"  {DM}·{R}  {color}{text}{R}"

def to_class_name(name):
    parts = re.split(r"[\s_\-]+", name)
    return "".join(p.capitalize() for p in parts if p)


# ── Code generators ────────────────────────────────────────────────────────────
def generate_agent(obs_size, shared_layers, policy_layers):
    return f"""\
import os
import numpy as np
import torch
from discrete_policy import DiscreteFF
from your_act import LookupAction

OBS_SIZE = {obs_size}

SHARED_LAYER_SIZES = {repr(shared_layers)}
POLICY_LAYER_SIZES = {repr(policy_layers)}


class Agent:
    def __init__(self):
        self.action_parser = LookupAction()
        self.num_actions = len(self.action_parser._lookup_table)
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        device = torch.device("cpu")

        self.policy = DiscreteFF(
            OBS_SIZE, self.num_actions,
            SHARED_LAYER_SIZES, POLICY_LAYER_SIZES,
            device
        )

        shared_head_path = os.path.join(cur_dir, "SHARED_HEAD.LT")
        policy_path = os.path.join(cur_dir, "POLICY.LT")

        self.policy.shared_head.load_state_dict(
            torch.load(shared_head_path, map_location=device, weights_only=False).state_dict()
        )
        self.policy.policy.load_state_dict(
            torch.load(policy_path, map_location=device, weights_only=False).state_dict()
        )

        torch.set_num_threads(1)

    def act(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            action_idx, _ = self.policy.get_action(state, True)
        action = np.array(self.action_parser.parse_actions([action_idx]))
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 1:
            raise Exception("Invalid action:", action)
        return action
"""


def generate_bot(class_name, obs_type):
    return f"""\
import numpy as np
import os
import torch

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util import common_values
from util.game_state import GameState
from util.player_data import PlayerData

from agent import Agent as Agent_Base
from obs import {obs_type}


class {class_name}(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.base_agent = Agent_Base()
        self.base_obs_builder = {obs_type}()

        self.obs_mean = None
        self.obs_std = None

        self.tick_skip = 8
        self.game_state: GameState = None
        self.controls = SimpleControllerState()
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0

    def is_hot_reload_enabled(self):
        return True

    def initialize_agent(self, field_info=None):
        if field_info is None:
            field_info = self.get_field_info()
        self.field_info = field_info
        self.game_state = GameState(self.field_info)
        self.update_action = True
        self.ticks = self.tick_skip
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action and len(self.game_state.players) > self.index:
            self.update_action = False
            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]
            self.game_state.players = [player] + teammates + opponents

            obs = self.base_obs_builder.build_obs(player, self.game_state, self.action)
            if self.obs_mean is not None and self.obs_std is not None:
                obs = (obs - self.obs_mean) / self.obs_std
            self.action = self.base_agent.act(obs)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
"""


def generate_cfg(bot_name):
    return f"""\
[Locations]
looks_config = ./appearance.cfg
python_file = ./bot.py
requirements_file = ./requirements.txt

name = {bot_name}
maximum_tick_rate_preference = 120

[Details]
developer =
description =
language = Python
"""


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cls()
    banner()

    # ── Bot name ───────────────────────────────────────────────────────────────
    section("BOT NAME")
    bot_name   = ask("name")
    class_name = to_class_name(bot_name)
    print(tag(f"class  →  {class_name}"))

    # ── Obs type ───────────────────────────────────────────────────────────────
    section("OBS TYPE")
    print(f"  {DM}  1  ·  AdvancedObs         ( 1v1 or 2v2 or 3v3 ){R}")
    print(f"  {DM}  2  ·  AdvancedObsPadded   (     MultiMode     ){R}")
    print()
    obs_choice = ask("select", validator=lambda x: x in ("1", "2"), hint="· 1 or 2")
    obs_type   = "AdvancedObs" if obs_choice == "1" else "AdvancedObsPadded"
    print(tag(obs_type))

    # ── OBS size ───────────────────────────────────────────────────────────────
    section("OBS SIZE")
    obs_size = int(ask("size", validator=lambda x: x.isdigit() and int(x) > 0, hint="· 109 / 167 / 225"))

    # ── Architecture ───────────────────────────────────────────────────────────
    section("NETWORK ARCHITECTURE")
    shared_layers = ask_layers("shared layers")
    policy_layers = ask_layers("policy layers")

    # ── Build ──────────────────────────────────────────────────────────────────
    section("BUILDING")
    out_dir = os.path.join(SCRIPT_DIR, bot_name)

    if os.path.exists(out_dir):
        print(f"\n  {YL}  folder '{bot_name}' already exists{R}")
        ow = ask("overwrite?", validator=lambda x: x.lower() in ("y", "n"), hint="· y / n")
        if ow.lower() != "y":
            print(f"\n  {RD}  aborted{R}\n")
            return
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    files_to_copy = STATIC_FILES + ["obs.py"]
    for item in files_to_copy:
        src = os.path.join(ASSETS_DIR, item)
        dst = os.path.join(out_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
        print(f"  {GR}  +{R}  {DM}{item}{R}")

    with open(os.path.join(out_dir, "agent.py"), "w") as f:
        f.write(generate_agent(obs_size, shared_layers, policy_layers))
    print(f"  {GR}  +{R}  {DM}agent.py{R}")

    with open(os.path.join(out_dir, "bot.py"), "w") as f:
        f.write(generate_bot(class_name, obs_type))
    print(f"  {GR}  +{R}  {DM}bot.py{R}")

    with open(os.path.join(out_dir, "bot.cfg"), "w") as f:
        f.write(generate_cfg(bot_name))
    print(f"  {GR}  +{R}  {DM}bot.cfg{R}")

    # ── Summary ────────────────────────────────────────────────────────────────
    W = 46
    def row(k, v):
        v = str(v)
        if len(v) > 28: v = v[:25] + "..."
        print(f"  {BL}│{R}  {DM}{k:<10}{R}  {CY}{v:<28}{R}  {BL}│{R}")

    print(f"\n  {BL}╔{'═' * W}╗{R}")
    print(f"  {BL}║{R}  {BD}{WHT}{'DONE':^{W-2}}{R}  {BL}║{R}")
    print(f"  {BL}╠{'═' * W}╣{R}")
    row("name",     bot_name)
    row("class",    class_name)
    row("obs",      obs_type)
    row("obs size", obs_size)
    row("shared",   shared_layers)
    row("policy",   policy_layers)
    print(f"  {BL}╚{'═' * W}╝{R}")
    print(f"\n  {YL}  drop {BD}SHARED_HEAD.LT{R}{YL} + {BD}POLICY.LT{R}{YL} into  {DM}{bot_name}/{R}\n")


if __name__ == "__main__":
    main()

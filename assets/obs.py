import numpy as np
from typing import List

from util import common_values
from util.game_state import GameState
from util.player_data import PlayerData
from util.physics_object import PhysicsObject


class AdvancedObs:
    POS_COEF = 1.0 / 5000.0
    VEL_COEF = 1.0 / 2300.0
    ANG_VEL_COEF = 1.0 / 3.0

    def _add_player_to_obs(self, out: List[float], player: PlayerData, inverted: bool, ball_phys: PhysicsObject):
        car = player.inverted_car_data if inverted else player.car_data

        rot = car.rotation_mtx()  # columns: [forward, left, up]
        forward = rot[:, 0]
        up = rot[:, 2]

        local_ang = rot.T @ car.angular_velocity
        local_rel_pos = rot.T @ (ball_phys.position - car.position)
        local_rel_vel = rot.T @ (ball_phys.linear_velocity - car.linear_velocity)

        out += (car.position * self.POS_COEF).tolist()
        out += forward.tolist()
        out += up.tolist()
        out += (car.linear_velocity * self.VEL_COEF).tolist()
        out += (car.angular_velocity * self.ANG_VEL_COEF).tolist()
        out += (local_ang * self.ANG_VEL_COEF).tolist()

        out += (local_rel_pos * self.POS_COEF).tolist()
        out += (local_rel_vel * self.VEL_COEF).tolist()

        out.append(float(player.boost_amount))
        out.append(1.0 if player.on_ground else 0.0)
        out.append(1.0 if (player.has_flip or player.has_jump) else 0.0)
        out.append(1.0 if player.is_demoed else 0.0)
        out.append(1.0 if getattr(player, "has_jumped", False) else 0.0)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        obs: List[float] = []

        inverted = (player.team_num == common_values.ORANGE_TEAM)
        ball = state.inverted_ball if inverted else state.ball

        obs += (ball.position * self.POS_COEF).tolist()
        obs += (ball.linear_velocity * self.VEL_COEF).tolist()
        obs += (ball.angular_velocity * self.ANG_VEL_COEF).tolist()

        prev = np.asarray(previous_action, dtype=np.float32).reshape(-1)
        if prev.size != 8:
            if prev.size < 8:
                prev = np.pad(prev, (0, 8 - prev.size))
            else:
                prev = prev[:8]
        obs += prev.tolist()

        pads = state.get_boost_pads(inverted)
        timers = state.get_boost_pad_timers(inverted)
        for i in range(pads.shape[0]):
            if pads[i] >= 0.5:
                obs.append(1.0)
            else:
                obs.append(1.0 / (1.0 + float(timers[i])))

        self._add_player_to_obs(obs, player, inverted, ball)

        # selection logic:
        opponents = [p for p in state.players if p.team_num != player.team_num and p.car_id != player.car_id]
        if len(opponents) == 0:
            # opponents, fill with zeros
            obs += [0.0] * 29
        elif len(opponents) == 1:
            # opponent, default behavior
            self._add_player_to_obs(obs, opponents[0], inverted, ball)
        else:
            # opponents: choose closest opponent to the ball
            closest_opponent = min(opponents, key=lambda p: np.linalg.norm(p.car_data.position - ball.position))
            self._add_player_to_obs(obs, closest_opponent, inverted, ball)

        if len(obs) != 109:
            if len(obs) > 109:
                obs = obs[:109]
            else:
                obs.extend([0.0] * (109 - len(obs)))

        return np.asarray(obs, dtype=np.float32)


class AdvancedObsPadded(AdvancedObs):
    BALL_MAX_SPEED = 6000.0
    MAX_TEAMMATES  = 2
    MAX_OPPONENTS  = 3
    PLAYER_FEAT_SIZE = 29

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        obs: List[float] = []

        inverted = (player.team_num == common_values.ORANGE_TEAM)
        ball = state.inverted_ball if inverted else state.ball

        # Ball — matches C++: vel uses BALL_MAX_SPEED, not VEL_COEF
        obs += (ball.position * self.POS_COEF).tolist()
        obs += (ball.linear_velocity / self.BALL_MAX_SPEED).tolist()
        obs += (ball.angular_velocity * self.ANG_VEL_COEF).tolist()

        prev = np.asarray(previous_action, dtype=np.float32).reshape(-1)
        if prev.size != 8:
            prev = np.pad(prev, (0, 8 - prev.size)) if prev.size < 8 else prev[:8]
        obs += prev.tolist()

        pads = state.get_boost_pads(inverted)
        timers = state.get_boost_pad_timers(inverted)
        for i in range(pads.shape[0]):
            if pads[i] >= 0.5:
                obs.append(1.0)
            else:
                obs.append(1.0 / (1.0 + float(timers[i])))

        # Self
        self._add_player_to_obs(obs, player, inverted, ball)

        # Teammates and opponents — collected in state order, then zero-padded
        teammates: List[float] = []
        opponents: List[float] = []
        for other in state.players:
            if other.car_id == player.car_id:
                continue
            if other.team_num == player.team_num:
                self._add_player_to_obs(teammates, other, inverted, ball)
            else:
                self._add_player_to_obs(opponents, other, inverted, ball)

        # Pad teammates to MAX_TEAMMATES
        cur_tm = len(teammates) // self.PLAYER_FEAT_SIZE
        obs += teammates[:self.MAX_TEAMMATES * self.PLAYER_FEAT_SIZE]
        obs += [0.0] * (max(0, self.MAX_TEAMMATES - cur_tm) * self.PLAYER_FEAT_SIZE)

        # Pad opponents to MAX_OPPONENTS
        cur_op = len(opponents) // self.PLAYER_FEAT_SIZE
        obs += opponents[:self.MAX_OPPONENTS * self.PLAYER_FEAT_SIZE]
        obs += [0.0] * (max(0, self.MAX_OPPONENTS - cur_op) * self.PLAYER_FEAT_SIZE)

        return np.asarray(obs, dtype=np.float32)

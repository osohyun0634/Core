#!/usr/bin/env python3
"""
정밀 도킹 상태 머신.
goal_pid.py의 ROTATE/MOVE(단일축 + 대각선) 로직을 그대로 가져오되,
"영원히 도는 노드"가 아니라 "한 세트의 웨이포인트를 끝내면 done=True를 반환하는
재사용 가능한 클래스"로 감쌌다. Nav2가 approach point까지 데려다준 뒤
이 클래스가 넘겨받아 후진+정밀 yaw 정렬을 수행한다.

[속도 급변 수정]
기존엔 상태(ROTATE/MOVE)가 바뀌거나 웨이포인트가 끝날 때마다 그 틱에서 바로
목표 속도로 뛰어버렸음 (예: MOVE 중 전진하다가 다음 웨이포인트가 ROTATE면
선속도가 그 즉시 0으로 뚝 끊김). 특히 Nav2 주행 중 capture radius 안으로 들어와
도킹으로 갓 전환된 시점엔 로봇이 실제로 어느 정도 속도로 움직이고 있는 상태라
이 문제가 두드러짐.
-> step() 내부는 이제 "이번 틱에 내고 싶은 목표 속도(target_lin/target_ang)"만
   계산하고, 실제로 반환하는 cmd는 함수 맨 끝에서 현재 실측 속도(odom 기반
   current_lin_vel/current_ang_vel)를 기준으로 max_lin_accel/max_ang_accel만큼만
   변하도록 제한(slew-rate limit)한다. 상태 전환이 일어나도 물리적으로 급가감속이
   불가능한 만큼만 명령이 바뀌므로 모든 전환이 자연스럽게 이어진다.
"""
import math
from geometry_msgs.msg import Twist


class DockingStateMachine:
    """
    waypoints: goal_pid.py와 동일한 포맷
        (x, y, target_yaw[rad], move_type, [wait_seconds])
        move_type: 'MOVE_FORWARD' / 'MOVE_BACKWARD' / 'ROTATE' / 'MOVE_DIAGONAL' / 'WAIT'
    params: dict, goal_pid.py의 declare_parameter 목록과 동일한 키 +
        'max_lin_accel'(m/s^2), 'max_ang_accel'(rad/s^2) - 속도 변화율 제한값
    """

    def __init__(self, waypoints, params, logger, clock):
        self.waypoints = waypoints
        self.p = params
        self.logger = logger
        self.clock = clock
        self.reset()

    def reset(self):
        self.index = 0
        self.state = 'ROTATE'
        self.prev_signed_error = None
        self.prev_distance = None
        self.prev_move_time = None
        self.diag_heading_vec = None
        self.wait_start_time = None
        self.done = False
        # 속도 제한 계산용 이전 틱 시각 (control loop 주기가 살짝 흔들려도 정확한 dt 사용)
        self._prev_tick_time = None

    def _finish_waypoint(self):
        self.index += 1
        self.state = 'ROTATE'
        self.prev_signed_error = None
        self.prev_distance = None
        self.prev_move_time = None
        self.diag_heading_vec = None

    def _rate_limit(self, target_lin, target_ang, current_lin_vel, current_ang_vel):
        """실측 속도(odom) 기준으로 max_lin_accel/max_ang_accel만큼만 변하도록 제한."""
        now = self.clock.now()
        if self._prev_tick_time is None:
            dt = 0.05
        else:
            dt = max((now - self._prev_tick_time).nanoseconds / 1e9, 0.001)
        self._prev_tick_time = now

        max_dlin = self.p.get('max_lin_accel', 0.3) * dt
        max_dang = self.p.get('max_ang_accel', 2.0) * dt

        lin_diff = target_lin - current_lin_vel
        lin_diff = max(-max_dlin, min(max_dlin, lin_diff))
        limited_lin = current_lin_vel + lin_diff

        ang_diff = target_ang - current_ang_vel
        ang_diff = max(-max_dang, min(max_dang, ang_diff))
        limited_ang = current_ang_vel + ang_diff

        return limited_lin, limited_ang

    def step(self, current_x, current_y, current_yaw, current_lin_vel, current_ang_vel):
        """control loop 1틱 실행. (Twist, done:bool) 반환."""
        target_lin = 0.0
        target_ang = 0.0
        done = False

        if self.done or self.index >= len(self.waypoints):
            self.done = True
            # 종료 시에도 속도 0으로 즉시 끊지 않고 제한된 감속으로 수렴
            lin, ang = self._rate_limit(0.0, 0.0, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, True

        wp = self.waypoints[self.index]
        goal_x, goal_y, goal_yaw, move_type = wp[0], wp[1], wp[2], wp[3]
        wait_seconds = wp[4] if len(wp) > 4 else 0.0

        if move_type == 'WAIT':
            if self.wait_start_time is None:
                self.wait_start_time = self.clock.now()
                self.logger.info(f'도킹 웨이포인트 {self.index} WAIT 시작 ({wait_seconds:.1f}초)')
            elapsed = (self.clock.now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= wait_seconds:
                self.index += 1
                self.state = 'ROTATE'
                self.wait_start_time = None
            # target_lin/ang = 0 유지, 감속은 rate_limit이 알아서 처리
            lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, False

        is_move_waypoint = move_type in ('MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_DIAGONAL')

        if move_type == 'MOVE_DIAGONAL':
            dx0 = goal_x - current_x
            dy0 = goal_y - current_y
            effective_target_yaw = math.atan2(dy0, dx0)
        else:
            effective_target_yaw = goal_yaw

        if self.state == 'ROTATE':
            yaw_error = math.atan2(
                math.sin(effective_target_yaw - current_yaw),
                math.cos(effective_target_yaw - current_yaw))
            yaw_error_deg = math.degrees(abs(yaw_error))

            finished_rotate = False
            if is_move_waypoint:
                if yaw_error_deg < self.p['move_start_yaw_threshold_deg']:
                    self.logger.info(f'도킹 웨이포인트 {self.index} {move_type}, 헤딩 정렬 확인, 이동 시작')
                    self.state = 'MOVE'
                    self.prev_signed_error = None
                    self.prev_distance = None
                    self.prev_move_time = None
                    if move_type == 'MOVE_DIAGONAL':
                        dxs = goal_x - current_x
                        dys = goal_y - current_y
                        norm = math.hypot(dxs, dys)
                        self.diag_heading_vec = (1.0, 0.0) if norm < 1e-6 else (dxs / norm, dys / norm)
                    finished_rotate = True
            else:
                if abs(yaw_error) < self.p['yaw_tolerance']:
                    self.logger.info(f'도킹 웨이포인트 {self.index} 방향 정렬 완료')
                    self.index += 1
                    self.state = 'ROTATE'
                    finished_rotate = True

            if not finished_rotate:
                ang_z = self.p['kp_angle'] * yaw_error
                if abs(ang_z) < self.p['min_ang_vel']:
                    ang_z = self.p['min_ang_vel'] if ang_z > 0 else -self.p['min_ang_vel']
                target_ang = max(-self.p['max_ang'], min(self.p['max_ang'], ang_z))
            # finished_rotate인 틱은 target_lin/ang = 0 그대로 -> rate_limit이 부드럽게 처리

            lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, False

        elif self.state == 'MOVE':
            dx = goal_x - current_x
            dy = goal_y - current_y

            if move_type == 'MOVE_DIAGONAL':
                distance = math.hypot(dx, dy)
                hx, hy = self.diag_heading_vec if self.diag_heading_vec else (1.0, 0.0)
                projection = dx * hx + dy * hy

                if distance < self.p['goal_tolerance'] or projection <= 0.0:
                    self.logger.info(f'도킹 웨이포인트 {self.index} 위치 도착!')
                    self._finish_waypoint()
                    lin, ang = self._rate_limit(0.0, 0.0, current_lin_vel, current_ang_vel)
                    cmd = Twist()
                    cmd.linear.x = lin
                    cmd.angular.z = ang
                    return cmd, False

                now = self.clock.now()
                dt = 0.05 if self.prev_move_time is None else max(
                    (now - self.prev_move_time).nanoseconds / 1e9, 0.05)
                self.prev_move_time = now
                dist_deriv = 0.0 if self.prev_distance is None else (distance - self.prev_distance) / dt
                self.prev_distance = distance

                raw_vel = self.p['kp_dist'] * distance + self.p['kd_dist'] * dist_deriv
                target_lin = min(self.p['max_lin'], raw_vel)
                if target_lin < self.p['min_lin_vel']:
                    target_lin = self.p['min_lin_vel']

                live_heading = math.atan2(dy, dx)
                yaw_error = math.atan2(
                    math.sin(live_heading - current_yaw), math.cos(live_heading - current_yaw))
                ang_z = self.p['kp_angle_diagonal'] * yaw_error
                target_ang = max(-self.p['max_ang_diagonal'], min(self.p['max_ang_diagonal'], ang_z))

                lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
                cmd = Twist()
                cmd.linear.x = lin
                cmd.angular.z = ang
                return cmd, False

            # MOVE_FORWARD / MOVE_BACKWARD (단일축)
            if abs(math.cos(goal_yaw)) >= abs(math.sin(goal_yaw)):
                signed_error = dx
            else:
                signed_error = dy
            distance = abs(signed_error)

            overshoot = False
            if self.prev_signed_error is not None and self.prev_signed_error != 0:
                if (self.prev_signed_error > 0) != (signed_error > 0):
                    overshoot = True

            if distance < self.p['goal_tolerance'] or overshoot:
                if overshoot:
                    self.logger.warn(
                        f'도킹 웨이포인트 {self.index} 목표 지나침 감지(오버슈트), '
                        f'잔여오차 {distance:.4f}m -> 도착 처리')
                else:
                    self.logger.info(f'도킹 웨이포인트 {self.index} 위치 도착!')
                self._finish_waypoint()
                lin, ang = self._rate_limit(0.0, 0.0, current_lin_vel, current_ang_vel)
                cmd = Twist()
                cmd.linear.x = lin
                cmd.angular.z = ang
                return cmd, False

            self.prev_signed_error = signed_error

            now = self.clock.now()
            dt = 0.05 if self.prev_move_time is None else max(
                (now - self.prev_move_time).nanoseconds / 1e9, 0.05)
            self.prev_move_time = now
            dist_deriv = 0.0 if self.prev_distance is None else (distance - self.prev_distance) / dt
            self.prev_distance = distance

            raw_vel = self.p['kp_dist'] * distance + self.p['kd_dist'] * dist_deriv
            raw_target_lin = min(self.p['max_lin'], raw_vel)
            if raw_target_lin < self.p['min_lin_vel']:
                raw_target_lin = self.p['min_lin_vel']
            target_lin = -raw_target_lin if move_type == 'MOVE_BACKWARD' else raw_target_lin

            yaw_error = math.atan2(math.sin(goal_yaw - current_yaw), math.cos(goal_yaw - current_yaw))
            yaw_error_deg = math.degrees(abs(yaw_error))
            if yaw_error_deg < self.p['move_yaw_deadzone_deg']:
                target_ang = 0.0
            else:
                correction = self.p['move_yaw_correction_gain'] * yaw_error
                target_ang = max(-self.p['move_yaw_correction_limit'],
                                  min(self.p['move_yaw_correction_limit'], correction))

            lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, False

        # 방어적 fallback (도달할 일 없음)
        lin, ang = self._rate_limit(0.0, 0.0, current_lin_vel, current_ang_vel)
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd, False
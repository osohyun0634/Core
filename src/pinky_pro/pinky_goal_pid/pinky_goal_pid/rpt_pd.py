#!/usr/bin/env python3
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
from rcl_interfaces.msg import SetParametersResult
import math

class GoalPID(Node):
    def __init__(self):
        super().__init__('rpt_pd')

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.debug_current_yaw_pub = self.create_publisher(Float64, '/debug/current_yaw', 10)
        self.debug_target_yaw_pub = self.create_publisher(Float64, '/debug/target_yaw', 10)
        self.debug_yaw_error_pub = self.create_publisher(Float64, '/debug/yaw_error', 10)

        self.debug_current_lin_vel_pub = self.create_publisher(Float64, '/debug/current_lin_vel', 10)
        self.debug_target_lin_vel_pub = self.create_publisher(Float64, '/debug/target_lin_vel', 10)
        self.debug_lin_vel_error_pub = self.create_publisher(Float64, '/debug/lin_vel_error', 10)

        self.debug_dist_error_pub = self.create_publisher(Float64, '/debug/dist_error', 10)
        self.debug_dist_error_deriv_pub = self.create_publisher(Float64, '/debug/dist_error_deriv', 10)
        self.debug_p_term_pub = self.create_publisher(Float64, '/debug/p_term', 10)
        self.debug_d_term_pub = self.create_publisher(Float64, '/debug/d_term', 10)

        # (x, y, target_yaw[rad], move_type)
        # move_type:
        #   'MOVE_FORWARD'  -> 전진만 (goal_yaw 방향 고정)
        #   'MOVE_BACKWARD' -> 후진만 (goal_yaw 방향 고정)
        #   'ROTATE'        -> 회전만
        #   'MOVE_DIAGONAL' -> 대각선 이동. target_yaw는 무시됨(자리채움용 아무 값이나 OK).
        #                      시작 시 (goal - current)로 헤딩을 자동 계산해서 그 방향으로
        #                      회전 후 전진하며, 이동 중에도 목표점을 계속 바라보도록 조향함.
        self.waypoints = [
            (1.433, 0.105, 3.138, 'MOVE_FORWARD'),
            (1.468, 0.101, -1.572, 'ROTATE'),
            (1.469, 0.495, -1.572, 'MOVE_BACKWARD'),
            (1.471, 0.503, 3.138, 'ROTATE'),
            (1.095, 0.500, 3.138, 'MOVE_FORWARD'),
            (1.095, 0.500, 3.138, 'WAIT', 5.0),
            
            (0.792, 0.454, -2.831, 'MOVE_DIAGONAL'),
            (0.297, 0.454, 3.138, 'MOVE_FORWARD'),
            (0.297, 0.454, 3.138, 'WAIT', 5.0),
            
            (0.152, 0.454, 3.138, 'MOVE_FORWARD'),
            (0.130, 0.444, -1.570, 'ROTATE'),
            (0.130, 0.035, -1.570, 'MOVE_FORWARD'),
            (0.152, 0.022, -0.003, 'ROTATE'),
            (0.788, 0.035, -0.003, 'MOVE_FORWARD'),
            
            (1.081, 0.293, 0.652, 'MOVE_DIAGONAL'),
            (1.321, 0.097, -0.658, 'MOVE_DIAGONAL'),
            
            (1.329, 0.056, 3.138, 'ROTATE'),
            (1.572, 0.057, 3.138,'MOVE_BACKWARD'),
        ]

        self.current_index = 0
        self.state = 'ROTATE'

        # 루프 모드: 정방향 웨이포인트 끝까지 가면 자동으로 시작점까지 복귀 후 재시작.
        # PD 게인 튜닝할 때 노드 재시작 없이 계속 왕복시키려고 추가.
        self.declare_parameter('loop_mode', True)
        self.loop_mode = self.get_parameter('loop_mode').value
        self.phase = 'FORWARD'  # 'FORWARD' 또는 'RETURN'
        self.return_waypoints = self._build_return_waypoints()

        self.declare_parameter('kp_angle', 0.6)
        self.declare_parameter('kp_dist', 0.3)
        self.declare_parameter('kd_dist', 0.05)
        self.declare_parameter('max_lin', 0.08)
        self.declare_parameter('max_ang', 0.25)
        self.declare_parameter('goal_tolerance', 0.01)
        self.declare_parameter('yaw_tolerance', 0.02)
        self.declare_parameter('min_ang_vel', 0.1)
        self.declare_parameter('min_lin_vel', 0.008)
        self.declare_parameter('move_yaw_correction_gain', 0.3)
        self.declare_parameter('move_yaw_correction_limit', 0.0087)  # 1초에 0.5도
        self.declare_parameter('move_yaw_deadzone_deg', 0.5)
        self.declare_parameter('move_start_yaw_threshold_deg', 2.0)
        # 대각선 이동 전용 파라미터: 직선 이동의 move_yaw_correction_gain/limit보다
        # 훨씬 크게 잡아서 실제로 목표점 쪽으로 방향을 꺾어가며 이동하게 함
        self.declare_parameter('kp_angle_diagonal', 0.5)
        self.declare_parameter('max_ang_diagonal', 0.2)

        self.kp_angle = self.get_parameter('kp_angle').value
        self.kp_dist = self.get_parameter('kp_dist').value
        self.kd_dist = self.get_parameter('kd_dist').value
        self.max_lin = self.get_parameter('max_lin').value
        self.max_ang = self.get_parameter('max_ang').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        self.min_ang_vel = self.get_parameter('min_ang_vel').value
        self.min_lin_vel = self.get_parameter('min_lin_vel').value
        self.move_yaw_correction_gain = self.get_parameter('move_yaw_correction_gain').value
        self.move_yaw_correction_limit = self.get_parameter('move_yaw_correction_limit').value
        self.move_yaw_deadzone_deg = self.get_parameter('move_yaw_deadzone_deg').value
        self.move_start_yaw_threshold_deg = self.get_parameter('move_start_yaw_threshold_deg').value
        self.kp_angle_diagonal = self.get_parameter('kp_angle_diagonal').value
        self.max_ang_diagonal = self.get_parameter('max_ang_diagonal').value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0

        self.prev_signed_error = None
        self.prev_distance = None
        self.prev_move_time = None

        # 대각선 이동 전용 상태: 진입 시점에 계산한 헤딩 방향 벡터(고정)
        # -> 오버슈트 판정(투영값 부호)에 사용
        self.diag_heading_vec = None

        # WAIT 상태 전용: 대기 시작 시각. None이면 아직 대기 시작 전.
        self.wait_start_time = None

        self.timer = self.create_timer(0.05, self.control_loop)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'kp_angle':
                self.kp_angle = param.value
            elif param.name == 'kp_dist':
                self.kp_dist = param.value
            elif param.name == 'kd_dist':
                self.kd_dist = param.value
            elif param.name == 'max_lin':
                self.max_lin = param.value
            elif param.name == 'max_ang':
                self.max_ang = param.value
            elif param.name == 'goal_tolerance':
                self.goal_tolerance = param.value
            elif param.name == 'yaw_tolerance':
                self.yaw_tolerance = param.value
            elif param.name == 'min_ang_vel':
                self.min_ang_vel = param.value
            elif param.name == 'min_lin_vel':
                self.min_lin_vel = param.value
            elif param.name == 'move_yaw_correction_gain':
                self.move_yaw_correction_gain = param.value
            elif param.name == 'move_yaw_correction_limit':
                self.move_yaw_correction_limit = param.value
            elif param.name == 'move_yaw_deadzone_deg':
                self.move_yaw_deadzone_deg = param.value
            elif param.name == 'move_start_yaw_threshold_deg':
                self.move_start_yaw_threshold_deg = param.value
            elif param.name == 'kp_angle_diagonal':
                self.kp_angle_diagonal = param.value
            elif param.name == 'max_ang_diagonal':
                self.max_ang_diagonal = param.value
            elif param.name == 'loop_mode':
                self.loop_mode = param.value
            self.get_logger().info(f'{param.name} 변경: {param.value}')
        return SetParametersResult(successful=True)

    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    def odom_callback(self, msg):
        self.current_lin_vel = msg.twist.twist.linear.x
        self.current_ang_vel = msg.twist.twist.angular.z

    def is_wait_condition_met(self, elapsed_sec, wait_seconds):
        # 지금은 고정 시간 대기. 나중에 Jetcobot 완료 신호로 바꿀 때는
        # 이 함수 안만 아래처럼 교체하면 됨 (구독 콜백에서 플래그 세팅 후 여기서 확인):
        #   return self.arm_done_flag
        return elapsed_sec >= wait_seconds

    def _build_return_waypoints(self):
        # 정방향 웨이포인트들의 (x, y)만 순서대로 뽑아서 연속 중복 좌표를 제거하고
        # (WAIT처럼 같은 자리를 두 번 찍는 경우 대비), 역순으로 되짚어가는
        # MOVE_DIAGONAL 시퀀스를 만든다. MOVE_DIAGONAL은 헤딩을 실시간 계산하므로
        # 원래 move_type이 뭐였든 신경 쓸 필요 없이 점만 순서대로 찍어주면 된다.
        coords = []
        for wp in self.waypoints:
            x, y = wp[0], wp[1]
            if not coords or abs(coords[-1][0] - x) > 1e-6 or abs(coords[-1][1] - y) > 1e-6:
                coords.append((x, y))

        if len(coords) < 2:
            return []

        reversed_coords = list(reversed(coords))[1:]  # 현재 끝점 자신은 제외
        return [(x, y, 0.0, 'MOVE_DIAGONAL') for x, y in reversed_coords]

    def control_loop(self):
        if self.current_x is None:
            return

        active_waypoints = self.waypoints if self.phase == 'FORWARD' else self.return_waypoints

        if self.current_index >= len(active_waypoints):
            if self.loop_mode and len(self.return_waypoints) > 0:
                if self.phase == 'FORWARD':
                    self.get_logger().info('정방향 웨이포인트 완료 -> 복귀 시작')
                    self.phase = 'RETURN'
                else:
                    self.get_logger().info('복귀 완료 -> 루프 재시작')
                    self.phase = 'FORWARD'
                self.current_index = 0
                self.state = 'ROTATE'
                self.pub.publish(Twist())
                return
            self.pub.publish(Twist())
            return

        waypoint = active_waypoints[self.current_index]
        goal_x, goal_y, goal_yaw, move_type = waypoint[0], waypoint[1], waypoint[2], waypoint[3]
        wait_seconds = waypoint[4] if len(waypoint) > 4 else 0.0
        cmd = Twist()

        if move_type == 'WAIT':
            self.pub.publish(Twist())  # 정지 유지

            if self.wait_start_time is None:
                self.wait_start_time = self.get_clock().now()
                self.get_logger().info(
                    f'웨이포인트 {self.current_index} WAIT 시작 ({wait_seconds:.1f}초)')

            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9

            if self.is_wait_condition_met(elapsed, wait_seconds):
                self.get_logger().info(f'웨이포인트 {self.current_index} WAIT 종료, 이동 재개')
                self.current_index += 1
                self.state = 'ROTATE'
                self.wait_start_time = None
            return

        is_move_waypoint = move_type in ('MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_DIAGONAL')

        # 대각선 웨이포인트는 저장된 goal_yaw 대신 (goal - current)로 계산한
        # 실시간 헤딩을 목표 각도로 사용한다.
        if move_type == 'MOVE_DIAGONAL':
            dx0 = goal_x - self.current_x
            dy0 = goal_y - self.current_y
            effective_target_yaw = math.atan2(dy0, dx0)
        else:
            effective_target_yaw = goal_yaw

        if self.state == 'ROTATE':
            yaw_error = effective_target_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            yaw_error_deg = math.degrees(abs(yaw_error))

            self.debug_current_yaw_pub.publish(Float64(data=self.current_yaw))
            self.debug_target_yaw_pub.publish(Float64(data=effective_target_yaw))
            self.debug_yaw_error_pub.publish(Float64(data=yaw_error))

            if is_move_waypoint:
                if yaw_error_deg < self.move_start_yaw_threshold_deg:
                    self.get_logger().info(
                        f'웨이포인트 {self.current_index} {move_type}, '
                        f'헤딩 정렬 확인({yaw_error_deg:.2f}도) 이동 시작...')
                    self.state = 'MOVE'
                    self.prev_signed_error = None
                    self.prev_distance = None
                    self.prev_move_time = None
                    if move_type == 'MOVE_DIAGONAL':
                        # MOVE 진입 시점의 헤딩을 고정해서 오버슈트 투영 기준으로 사용
                        dxs = goal_x - self.current_x
                        dys = goal_y - self.current_y
                        norm = math.hypot(dxs, dys)
                        if norm < 1e-6:
                            self.diag_heading_vec = (1.0, 0.0)
                        else:
                            self.diag_heading_vec = (dxs / norm, dys / norm)
                    self.pub.publish(Twist())
                    return
                # else: 오차 큼 -> 아래 회전 제어로 진행
            else:
                if abs(yaw_error) < self.yaw_tolerance:
                    self.get_logger().info(f'웨이포인트 {self.current_index} 방향 정렬 완료')
                    self.current_index += 1
                    self.state = 'ROTATE'
                    self.pub.publish(Twist())
                    return

            cmd.linear.x = 0.0
            ang_z = self.kp_angle * yaw_error
            if abs(ang_z) < self.min_ang_vel:
                ang_z = self.min_ang_vel if ang_z > 0 else -self.min_ang_vel
            cmd.angular.z = max(-self.max_ang, min(self.max_ang, ang_z))

            self.debug_target_lin_vel_pub.publish(Float64(data=cmd.angular.z))
            self.debug_current_lin_vel_pub.publish(Float64(data=self.current_ang_vel))
            self.debug_lin_vel_error_pub.publish(Float64(data=cmd.angular.z - self.current_ang_vel))

            self.pub.publish(cmd)

        elif self.state == 'MOVE':
            dx = goal_x - self.current_x
            dy = goal_y - self.current_y

            if move_type == 'MOVE_DIAGONAL':
                # --- 대각선 전용 처리 ---
                distance = math.hypot(dx, dy)
                self.debug_dist_error_pub.publish(Float64(data=distance))

                # 오버슈트/도착 판정: 진입 시 고정한 헤딩 벡터에 잔여벡터를 투영.
                # 투영값이 0 이하가 되면 목표점을 지나쳤거나 도착한 것으로 간주.
                hx, hy = self.diag_heading_vec if self.diag_heading_vec else (1.0, 0.0)
                projection = dx * hx + dy * hy

                if distance < self.goal_tolerance or projection <= 0.0:
                    if projection <= 0.0 and distance >= self.goal_tolerance:
                        self.get_logger().warn(
                            f'웨이포인트 {self.current_index} 목표 지나침 감지(대각선 오버슈트), '
                            f'잔여오차 {distance:.4f}m -> 도착 처리')
                    else:
                        self.get_logger().info(f'웨이포인트 {self.current_index} 위치 도착!')
                    self.current_index += 1
                    self.state = 'ROTATE'
                    self.prev_signed_error = None
                    self.prev_distance = None
                    self.prev_move_time = None
                    self.diag_heading_vec = None
                    self.pub.publish(Twist())
                    return

                now = self.get_clock().now()
                if self.prev_move_time is None:
                    dt = 0.05
                else:
                    dt = (now - self.prev_move_time).nanoseconds / 1e9
                    if dt <= 0.0:
                        dt = 0.05
                self.prev_move_time = now

                if self.prev_distance is None:
                    dist_deriv = 0.0
                else:
                    dist_deriv = (distance - self.prev_distance) / dt
                self.prev_distance = distance

                p_term = self.kp_dist * distance
                d_term = self.kd_dist * dist_deriv
                target_lin_vel = p_term + d_term
                target_lin_vel = min(self.max_lin, target_lin_vel)
                if target_lin_vel < self.min_lin_vel:
                    target_lin_vel = self.min_lin_vel

                self.debug_dist_error_deriv_pub.publish(Float64(data=dist_deriv))
                self.debug_p_term_pub.publish(Float64(data=p_term))
                self.debug_d_term_pub.publish(Float64(data=d_term))

                # 대각선은 항상 목표를 향해 전진 (후진 대각선은 미지원)
                cmd.linear.x = target_lin_vel

                # 목표점을 계속 바라보도록 실시간으로 헤딩 재계산 후 강하게 조향
                live_heading = math.atan2(dy, dx)
                yaw_error = live_heading - self.current_yaw
                yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
                ang_z = self.kp_angle_diagonal * yaw_error
                cmd.angular.z = max(-self.max_ang_diagonal, min(self.max_ang_diagonal, ang_z))

                self.debug_current_yaw_pub.publish(Float64(data=self.current_yaw))
                self.debug_target_yaw_pub.publish(Float64(data=live_heading))
                self.debug_yaw_error_pub.publish(Float64(data=yaw_error))

                self.debug_target_lin_vel_pub.publish(Float64(data=cmd.linear.x))
                self.debug_current_lin_vel_pub.publish(Float64(data=self.current_lin_vel))
                self.debug_lin_vel_error_pub.publish(Float64(data=cmd.linear.x - self.current_lin_vel))

                self.pub.publish(cmd)
                return

            # --- 기존 MOVE_FORWARD / MOVE_BACKWARD 처리 (단일축) ---
            if abs(math.cos(goal_yaw)) >= abs(math.sin(goal_yaw)):
                signed_error = dx
            else:
                signed_error = dy

            distance = abs(signed_error)
            self.debug_dist_error_pub.publish(Float64(data=distance))

            overshoot = False
            if self.prev_signed_error is not None and self.prev_signed_error != 0:
                if (self.prev_signed_error > 0) != (signed_error > 0):
                    overshoot = True

            if distance < self.goal_tolerance or overshoot:
                if overshoot:
                    self.get_logger().warn(
                        f'웨이포인트 {self.current_index} 목표 지나침 감지(오버슈트), '
                        f'잔여오차 {distance:.4f}m -> 도착 처리')
                else:
                    self.get_logger().info(f'웨이포인트 {self.current_index} 위치 도착!')
                self.current_index += 1
                self.state = 'ROTATE'
                self.prev_signed_error = None
                self.prev_distance = None
                self.prev_move_time = None
                self.pub.publish(Twist())
                return

            self.prev_signed_error = signed_error

            now = self.get_clock().now()
            if self.prev_move_time is None:
                dt = 0.05
            else:
                dt = (now - self.prev_move_time).nanoseconds / 1e9
                if dt <= 0.0:
                    dt = 0.05
            self.prev_move_time = now

            if self.prev_distance is None:
                dist_deriv = 0.0
            else:
                dist_deriv = (distance - self.prev_distance) / dt
            self.prev_distance = distance

            p_term = self.kp_dist * distance
            d_term = self.kd_dist * dist_deriv
            target_lin_vel = p_term + d_term

            target_lin_vel = min(self.max_lin, target_lin_vel)
            if target_lin_vel < self.min_lin_vel:
                target_lin_vel = self.min_lin_vel

            self.debug_dist_error_deriv_pub.publish(Float64(data=dist_deriv))
            self.debug_p_term_pub.publish(Float64(data=p_term))
            self.debug_d_term_pub.publish(Float64(data=d_term))

            if move_type == 'MOVE_BACKWARD':
                cmd.linear.x = -target_lin_vel
            else:
                cmd.linear.x = target_lin_vel

            yaw_error = goal_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            yaw_error_deg = math.degrees(abs(yaw_error))

            if yaw_error_deg < self.move_yaw_deadzone_deg:
                cmd.angular.z = 0.0
            else:
                correction = self.move_yaw_correction_gain * yaw_error
                correction = max(-self.move_yaw_correction_limit,
                                  min(self.move_yaw_correction_limit, correction))
                cmd.angular.z = correction

            self.debug_current_yaw_pub.publish(Float64(data=self.current_yaw))
            self.debug_target_yaw_pub.publish(Float64(data=goal_yaw))
            self.debug_yaw_error_pub.publish(Float64(data=yaw_error))

            self.debug_target_lin_vel_pub.publish(Float64(data=cmd.linear.x))
            self.debug_current_lin_vel_pub.publish(Float64(data=self.current_lin_vel))
            self.debug_lin_vel_error_pub.publish(Float64(data=cmd.linear.x - self.current_lin_vel))

            self.pub.publish(cmd)

def main():
    rclpy.init()
    node = GoalPID()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
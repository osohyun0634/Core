#!/usr/bin/env python3
"""
스테이션 리스트를 순서대로 순회하는 웨이포인트 매니저.

  1. Nav2 NavigateToPose로 approach point까지 이동 (장애물 회피, loose tolerance)
     - 단, 정밀 정차가 필요한 스테이션은 approach_pose까지 CAPTURE_RADIUS(10cm) 안으로
       들어오는 순간 Nav2 목표를 취소하고 곧바로 도킹 상태 머신으로 넘어감.
       Nav2의 목표 근처 감속/재가속(부자연스러운 속도 변화, 부정확한 도착)을 피하고
       더 이른 지점부터 도킹 컨트롤러의 예측 가능한 접근으로 이어받기 위함.
  2. 도킹 상태 머신(dock_control.py)이 ROTATE/MOVE 로직으로 후진 + 정밀 yaw 정렬 수행
     (Nav2는 이 구간 동안 관여하지 않음)
  3. 도킹 완료 후 WAIT(타이어 교체 대기)
  4. 다음 스테이션으로 반복, 마지막 스테이션 이후 처음으로 돌아가 루프(한 바퀴)

TODO:
  - is_wait_condition을 고정시간 대신 Jetcobot 완료 신호(토픽/서비스)로 교체할 경우
    WAITING 분기의 조건문만 바꾸면 됨
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from nav2_msgs.action import NavigateToPose

from pinky_goal_pid.dock_control import DockingStateMachine


# goal_pid.py에서 쓰던 값 그대로. 필요시 rqt_reconfigure로 붙여서 튜닝해도 됨.
DOCKING_PARAMS = {
    'kp_angle': 0.6,
    'kp_dist': 0.3,
    'kd_dist': 0.05,
    'max_lin': 0.08,
    'max_ang': 0.29,
    'goal_tolerance': 0.01,
    'yaw_tolerance': 0.018,
    'min_ang_vel': 0.1,
    'min_lin_vel': 0.008,
    'move_yaw_correction_gain': 0.3,
    'move_yaw_correction_limit': 0.0087,
    'move_yaw_deadzone_deg': 0.5,
    'move_start_yaw_threshold_deg': 2.0,
    'kp_angle_diagonal': 0.5,
    'max_ang_diagonal': 0.2,
}

# Nav2 목표 전송 관련 타이밍
STARTUP_DELAY_SEC = 3.0     # 노드 시작 후 첫 목표 전송까지 대기 (Nav2 lifecycle 안정화 시간)
GOAL_RETRY_DELAY_SEC = 1.0  # 목표 거부/서버 미응답 시 재시도 간격

# Nav2 주행 중 approach_pose까지 이 거리(m) 안으로 들어오면 도킹으로 조기 전환.
# 정밀 정차가 필요한 스테이션(docking_waypoints가 있는 경우)에만 적용됨.
CAPTURE_RADIUS = 0.10


class Station:
    def __init__(self, name, approach_pose, docking_waypoints=None, wait_seconds=0.0):
        """
        approach_pose: (x, y, yaw) - Nav2 목표. general_goal_checker(loose tolerance) 적용.
        docking_waypoints: 정밀 정차가 필요한 스테이션만 채움 (타이어 운반 정차 지점).
                            None이면 approach 도착 후 바로 다음 스테이션으로 넘어감.
                            goal_pid.py 포맷과 동일: (x, y, yaw, move_type, [wait_seconds])
        wait_seconds: 도킹 완료 후 대기시간(타이어 교체 등). docking_waypoints가 없으면 무시됨.
        """
        self.name = name
        self.approach_pose = approach_pose
        self.docking_waypoints = docking_waypoints
        self.wait_seconds = wait_seconds


class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.sub_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0

        self.stations = [
            Station('approach_tire_stop_1', approach_pose=(1.488, 0.515, 3.138),
                    docking_waypoints=[
                        (1.488, 0.515, 3.138, 'MOVE_DIAGONAL'),
                        (1.488, 0.515, -1.572, 'ROTATE'),
                    ],
                    wait_seconds=3.0),
            Station('approach_tire_stop_2', approach_pose=(1.488, 0.515, -1.572),
                    docking_waypoints=[
                        (1.488, 0.515, 3.138, 'ROTATE'),
                        (1.085, 0.515, 3.138, 'MOVE_FORWARD'),
                    ],
                    wait_seconds=3.0),
            Station('approach_tire_stop_3', approach_pose=(0.303, 0.472, 3.138),
                    docking_waypoints=[
                        (0.303, 0.468, 3.138, 'MOVE_FORWARD'),
                    ],
                    wait_seconds=3.0),
            Station('approach_tire_stop_4', approach_pose=(0.303, 0.468, 3.138),
                    docking_waypoints=[
                        (0.153, 0.468, 3.138, 'MOVE_FORWARD'),
                        (0.143, 0.454, -1.567, 'ROTATE'),
                        (0.133, 0.035, -1.567, 'MOVE_FORWARD'),
                        # (0.155, 0.022, -0.003, 'ROTATE'),
                        # (0.791, 0.035, -0.003, 'MOVE_FORWARD'),

                        # (1.083, 0.293, 0.652, 'MOVE_DIAGONAL'),
                        # (1.323, 0.097, -0.658, 'MOVE_DIAGONAL'),

                        # (1.332, 0.056, 3.138, 'ROTATE'),
                        # (1.574, 0.057, 3.138,'MOVE_BACKWARD'),
                    ],
                    wait_seconds=0.0),

            Station('return_to_start', approach_pose=(1.5625, 0.081, -3.138),
                    docking_waypoints=[
                        (1.5625, 0.071, -3.138, 'MOVE_DIAGONAL'),
                        (1.5425, 0.081, -3.138, 'ROTATE'),
                    ],
                    wait_seconds=1.0),
        ]

        self.station_index = 0
        self.docking_sm = None
        self.mode = 'NAV2'  # 'NAV2' -> 'DOCKING' -> 'WAITING'
        self.wait_start_time = None

        # Nav2 조기 취소/전환 관련 상태
        self.current_goal_handle = None
        self.docking_triggered_early = False

        # 재시도/재전송용 1회성 타이머 핸들. 여러 개 쌓이지 않도록 항상 취소 후 재생성.
        self._retry_timer = None

        self.declare_parameter('loop_mode', True)
        self.loop_mode = self.get_parameter('loop_mode').value

        self.timer = self.create_timer(0.05, self.control_loop)

        # Nav2 lifecycle(bt_navigator 등)이 active 상태로 안정화될 시간을 벌어준 뒤 시작.
        # wait_for_server는 액션 서버 '존재'만 확인하지 '목표 처리 준비'는 보장 안 하므로,
        # 초반에 바로 목표를 보내면 accept/reject 스팸이 발생할 수 있음.
        self.get_logger().info(f'{STARTUP_DELAY_SEC}초 후 첫 목표 전송...')
        self._retry_timer = self.create_timer(STARTUP_DELAY_SEC, self._start_once)

    def _start_once(self):
        self._cancel_retry_timer()
        self.send_next_nav2_goal()

    def _cancel_retry_timer(self):
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    def _schedule_retry(self, delay_sec, callback):
        self._cancel_retry_timer()
        self._retry_timer = self.create_timer(delay_sec, lambda: self._fire_once(callback))

    def _fire_once(self, callback):
        self._cancel_retry_timer()
        callback()

    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    def odom_callback(self, msg):
        self.current_lin_vel = msg.twist.twist.linear.x
        self.current_ang_vel = msg.twist.twist.angular.z

    # ---------- Nav2 구간 ----------
    def send_next_nav2_goal(self):
        station = self.stations[self.station_index]
        x, y, yaw = station.approach_pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(
            f'[{station.name}] Nav2 목표 전송: ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}도)')
        self.mode = 'NAV2'
        self.current_goal_handle = None
        self.docking_triggered_early = False

        if not self.nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'Nav2 액션 서버 응답 없음, {GOAL_RETRY_DELAY_SEC}초 후 재시도')
            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self.send_next_nav2_goal)
            return

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.nav2_goal_response_callback)

    def nav2_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                f'Nav2 목표 거부됨, {GOAL_RETRY_DELAY_SEC}초 후 재시도')
            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self.send_next_nav2_goal)
            return
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav2_result_callback)

    def nav2_result_callback(self, future):
        # 이미 capture radius 진입으로 도킹으로 조기 전환된 경우, 뒤늦게 도착하는
        # (취소됐거나 정상 완료된) Nav2 결과 콜백은 무시한다. 안 그러면 이미 DOCKING
        # 모드로 넘어간 뒤에 advance_to_next_station이 다시 불려서 스테이션이 씹힐 수 있음.
        if self.docking_triggered_early:
            return

        station = self.stations[self.station_index]
        self.get_logger().info(f'[{station.name}] Nav2 도착 완료')

        if station.docking_waypoints:
            self.get_logger().info(f'[{station.name}] 정밀 도킹 시작')
            self.docking_sm = DockingStateMachine(
                station.docking_waypoints, DOCKING_PARAMS, self.get_logger(), self.get_clock())
            self.mode = 'DOCKING'
        else:
            self.advance_to_next_station()

    def _trigger_early_docking(self):
        """NAV2 주행 중 approach_pose까지 CAPTURE_RADIUS 안에 들어왔을 때 호출.
        Nav2 목표를 취소하고 즉시 도킹 상태 머신으로 전환한다."""
        self.docking_triggered_early = True

        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()

        station = self.stations[self.station_index]
        self.get_logger().info(
            f'[{station.name}] Nav2 주행 중 {CAPTURE_RADIUS:.2f}m 이내 진입, '
            f'도킹으로 조기 전환')
        self.docking_sm = DockingStateMachine(
            station.docking_waypoints, DOCKING_PARAMS, self.get_logger(), self.get_clock())
        self.mode = 'DOCKING'

    # ---------- 도킹/대기/전환 구간 ----------
    def advance_to_next_station(self):
        self.station_index += 1
        if self.station_index >= len(self.stations):
            if self.loop_mode:
                self.get_logger().info('한 바퀴 완료 -> 루프 재시작')
                self.station_index = 0
            else:
                self.get_logger().info('전체 경로 완료')
                self.mode = 'DONE'
                return
        self.send_next_nav2_goal()

    def control_loop(self):
        if self.current_x is None:
            return

        if self.mode == 'NAV2':
            station = self.stations[self.station_index]
            # 정밀 정차가 필요한 스테이션만 조기 전환 로직 적용
            if station.docking_waypoints and not self.docking_triggered_early:
                ax, ay, _ = station.approach_pose
                dist = math.hypot(ax - self.current_x, ay - self.current_y)
                if dist < CAPTURE_RADIUS:
                    self._trigger_early_docking()
            # Nav2 controller_server가 /cmd_vel을 직접 발행 중이므로 여기서는 개입하지 않음
            return

        elif self.mode == 'DOCKING':
            cmd, done = self.docking_sm.step(
                self.current_x, self.current_y, self.current_yaw,
                self.current_lin_vel, self.current_ang_vel)
            self.cmd_pub.publish(cmd)
            if done:
                self.cmd_pub.publish(Twist())
                station = self.stations[self.station_index]
                self.get_logger().info(f'[{station.name}] 도킹 완료')
                if station.wait_seconds > 0:
                    self.wait_start_time = self.get_clock().now()
                    self.mode = 'WAITING'
                else:
                    self.advance_to_next_station()

        elif self.mode == 'WAITING':
            self.cmd_pub.publish(Twist())
            station = self.stations[self.station_index]
            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= station.wait_seconds:
                self.get_logger().info(f'[{station.name}] 대기 종료')
                self.advance_to_next_station()


def main():
    rclpy.init()
    node = WaypointManager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
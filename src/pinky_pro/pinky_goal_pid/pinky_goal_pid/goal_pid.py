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
        super().__init__('goal_pid')

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

        # 진행축 기준 거리 오차 디버그용
        self.debug_dist_error_pub = self.create_publisher(Float64, '/debug/dist_error', 10)

        # (x, y, target_yaw[rad], move_type)
        # move_type: 'MOVE_FORWARD' -> 전진만, 'MOVE_BACKWARD' -> 후진만, 'ROTATE' -> 회전만
        self.waypoints = [
            (1.433, 0.105, 3.138, 'MOVE_FORWARD'),
            (1.468, 0.101, -1.572, 'ROTATE'),
            (1.469, 0.438, -1.572, 'MOVE_BACKWARD'),
            (1.477, 0.432, 3.138, 'ROTATE'),
            (0.294, 0.434, 3.138, 'MOVE_FORWARD'),
        ]

        self.current_index = 0
        self.state = 'ROTATE'

        self.declare_parameter('kp_angle', 0.6)
        self.declare_parameter('kp_dist', 0.3)
        self.declare_parameter('max_lin', 0.08)
        self.declare_parameter('max_ang', 0.25)
        self.declare_parameter('goal_tolerance', 0.01)
        self.declare_parameter('yaw_tolerance', 0.02)
        self.declare_parameter('min_ang_vel', 0.1)
        self.declare_parameter('min_lin_vel', 0.008)
        self.declare_parameter('move_yaw_correction_gain', 0.3)
        self.declare_parameter('move_yaw_correction_limit', 0.0087)  # 1초에 0.5도
        self.declare_parameter('move_yaw_deadzone_deg', 0.5)          # 0.5도 미만이면 보정 안 함
        # MOVE_FORWARD/BACKWARD 진입 전 안전장치: 이 값(도) 이상 어긋나 있으면 먼저 회전하고 이동
        self.declare_parameter('move_start_yaw_threshold_deg', 2.0)
        # 위치 오차 1cm, 각도 오차 1.25도

        self.kp_angle = self.get_parameter('kp_angle').value
        self.kp_dist = self.get_parameter('kp_dist').value
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

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0

        # MOVE 상태에서 목표를 지나쳤는지(부호 반전) 감지하기 위한 이전 스텝의 부호 있는 오차
        self.prev_signed_error = None

        self.timer = self.create_timer(0.05, self.control_loop)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'kp_angle':
                self.kp_angle = param.value
            elif param.name == 'kp_dist':
                self.kp_dist = param.value
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

    def control_loop(self):
        if self.current_x is None:
            return

        if self.current_index >= len(self.waypoints):
            self.pub.publish(Twist())
            return

        goal_x, goal_y, goal_yaw, move_type = self.waypoints[self.current_index]
        cmd = Twist()

        if self.state == 'ROTATE':
            yaw_error = goal_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            yaw_error_deg = math.degrees(abs(yaw_error))

            self.debug_current_yaw_pub.publish(Float64(data=self.current_yaw))
            self.debug_target_yaw_pub.publish(Float64(data=goal_yaw))
            self.debug_yaw_error_pub.publish(Float64(data=yaw_error))

            is_move_waypoint = move_type in ('MOVE_FORWARD', 'MOVE_BACKWARD')

            if is_move_waypoint:
                # 안전장치: 오차가 threshold(기본 2도) 미만이면 회전 생략하고 바로 이동,
                # 그 이상이면 먼저 goal_yaw로 회전한 뒤 이동 (아래 회전 제어로 계속 진행)
                if yaw_error_deg < self.move_start_yaw_threshold_deg:
                    self.get_logger().info(
                        f'웨이포인트 {self.current_index} {move_type}, '
                        f'헤딩 정렬 확인({yaw_error_deg:.2f}도) 이동 시작...')
                    self.state = 'MOVE'
                    self.prev_signed_error = None
                    self.pub.publish(Twist())
                    return
                # else: 오차 큼 -> 아래 회전 제어로 진행 (index 증가 없이 이 웨이포인트 유지)
            else:
                # 순수 ROTATE 웨이포인트는 기존처럼 yaw_tolerance 기준으로 완료 판정
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

            # MOVE 상태는 진행방향(전/후진)만 제어하고 조향은 별도의 약한 yaw 보정만 하므로,
            # 도착 판정은 진행축(goal_yaw로 판별) 기준 단일축 오차로 계산.
            # goal_yaw가 0/180 근처면 x축 이동, ±90 근처면 y축 이동.
            if abs(math.cos(goal_yaw)) >= abs(math.sin(goal_yaw)):
                signed_error = dx
            else:
                signed_error = dy

            distance = abs(signed_error)
            self.debug_dist_error_pub.publish(Float64(data=distance))

            # 오버슈트 감지: amcl_pose 업데이트 간격이 넓어서 goal_tolerance 구간을
            # 샘플링으로 못 잡고 지나쳐버린 경우, 부호가 뒤집힌 걸로 감지해서 도착 처리
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
                self.pub.publish(Twist())
                return

            self.prev_signed_error = signed_error

            target_lin_vel = min(self.max_lin, self.kp_dist * distance)
            if target_lin_vel < self.min_lin_vel:
                target_lin_vel = self.min_lin_vel

            # move_type에 따라 부호만 고정 (판단 로직 없음)
            if move_type == 'MOVE_BACKWARD':
                cmd.linear.x = -target_lin_vel
            else:  # 'MOVE_FORWARD'
                cmd.linear.x = target_lin_vel

            # 목표 yaw와 0.5도 이상 벌어졌을 때만 아주 약하게 보정 (최대 0.5도/초)
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
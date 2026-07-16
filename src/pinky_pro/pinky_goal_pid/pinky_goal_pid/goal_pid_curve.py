from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf_transformations import euler_from_quaternion
from rcl_interfaces.msg import SetParametersResult
import math

# 주의: 이 버전은 MOVE 상태에서 선속도(전/후진)와 각속도(회전)를 "동시에" 제어합니다.
# 예전에 회전+이동을 섞었을 때 오실레이션(진동)이 생겨서 MOVE/ROTATE를 분리했던 이력이 있으니,
# 이 파일은 goal_pid.py(분리형)와 비교 실험용으로 쓰는 걸 권장합니다.

class GoalPID(Node):
    def __init__(self):
        super().__init__('goal_pid_curve')

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
        self.waypoints = [
            (1.433, 0.105, 3.138, 'MOVE_FORWARD'),
            (1.468, 0.101, -1.572, 'ROTATE'),
            (1.469, 0.448, -1.572, 'MOVE_BACKWARD'),
            (1.477, 0.442, 3.138, 'ROTATE'),
            (0.294, 0.444, 3.138, 'MOVE_FORWARD'),
        ]

        self.current_index = 0
        self.state = 'ROTATE'

        self.declare_parameter('kp_angle', 0.6)
        self.declare_parameter('kp_dist', 0.3)
        self.declare_parameter('kd_dist', 0.05)
        self.declare_parameter('max_lin', 0.08)
        self.declare_parameter('max_ang', 0.25)
        self.declare_parameter('goal_tolerance', 0.01)
        self.declare_parameter('yaw_tolerance', 0.02)
        self.declare_parameter('min_ang_vel', 0.1)
        self.declare_parameter('min_lin_vel', 0.008)
        # MOVE 중 회전 게인 (ROTATE 상태 kp_angle과 별개로 튜닝 가능하게 분리)
        self.declare_parameter('move_kp_angle', 0.3)
        # 위치 오차 1cm, 각도 오차 1.25도

        self.kp_angle = self.get_parameter('kp_angle').value
        self.kp_dist = self.get_parameter('kp_dist').value
        self.kd_dist = self.get_parameter('kd_dist').value
        self.max_lin = self.get_parameter('max_lin').value
        self.max_ang = self.get_parameter('max_ang').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        self.min_ang_vel = self.get_parameter('min_ang_vel').value
        self.min_lin_vel = self.get_parameter('min_lin_vel').value
        self.move_kp_angle = self.get_parameter('move_kp_angle').value

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0

        self.prev_signed_error = None
        self.prev_distance = None
        self.prev_move_time = None

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
            elif param.name == 'move_kp_angle':
                self.move_kp_angle = param.value
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
            # 순수 ROTATE 웨이포인트만 이 상태를 거침.
            # MOVE_FORWARD/BACKWARD는 회전 없이 바로 MOVE로 진입해서 이동+조향을 동시에 함.
            if move_type in ('MOVE_FORWARD', 'MOVE_BACKWARD'):
                self.get_logger().info(
                    f'웨이포인트 {self.current_index} {move_type}, 이동+조향 동시 제어 시작...')
                self.state = 'MOVE'
                self.prev_signed_error = None
                self.prev_distance = None
                self.prev_move_time = None
                self.pub.publish(Twist())
                return

            yaw_error = goal_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

            self.debug_current_yaw_pub.publish(Float64(data=self.current_yaw))
            self.debug_target_yaw_pub.publish(Float64(data=goal_yaw))
            self.debug_yaw_error_pub.publish(Float64(data=yaw_error))

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

            # 도착 판정용 거리는 그대로 진행축 기준 단일축 오차 사용 (오버슈트 감지 로직 유지)
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

            # --- 선속도: P + D (기존과 동일) ---
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

            # --- 각속도: ROTATE와 같은 비례제어를 "동시에" 적용 (핵심 차이점) ---
            yaw_error = goal_yaw - self.current_yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

            ang_z = self.move_kp_angle * yaw_error
            ang_z = max(-self.max_ang, min(self.max_ang, ang_z))
            cmd.angular.z = ang_z

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

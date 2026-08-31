#!/usr/bin/env python3


from __future__ import annotations

import argparse
import math
import struct
import sys

try:
    import serial
except ImportError:
    print("Warning: 'pyserial' is not installed. Running in simulation/mock mode.", file=sys.stderr)
    serial = None

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import tf2_ros


class ChassisSerialBridge(Node):
    def __init__(self, port: str = "/dev/ttyUSB_chassis", baudrate: int = 115200) -> None:
        super().__init__("chassis_serial_bridge")

        self.declare_parameter("port", port)
        self.declare_parameter("baudrate", baudrate)
        self.declare_parameter("send_interval_ms", 20)
        self.declare_parameter("wheel_radius", 0.076)
        self.declare_parameter("lx", 0.25)
        self.declare_parameter("ly", 0.25)
        self.declare_parameter("lift_lead", 0.01)
        self.declare_parameter("debug_mode", False)

        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.send_interval = (
            self.get_parameter("send_interval_ms").get_parameter_value().integer_value / 1000.0
        )
        self.wheel_radius = self.get_parameter("wheel_radius").get_parameter_value().double_value
        self.lx = self.get_parameter("lx").get_parameter_value().double_value
        self.ly = self.get_parameter("ly").get_parameter_value().double_value
        self.lift_lead = self.get_parameter("lift_lead").get_parameter_value().double_value
        self.debug_mode = self.get_parameter("debug_mode").get_parameter_value().boolean_value

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.cmd_lift_vel = 0.0

        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.lift_height = 0.0
        self.last_time = self.get_clock().now()

        self.ser = None
        if serial is not None:
            try:
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=0.05,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
                self.get_logger().info(
                    f"Opened chassis serial port {self.port} at {self.baudrate} baud."
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to open serial port {self.port}: {e}. Falling back to Mock mode."
                )
        else:
            self.get_logger().warn("Running in MOCK mode (serial library not found).")

        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)

        self.cmd_vel_sub = self.create_subscription(Twist, "cmd_vel", self._cmd_vel_callback, 10)
        self.lift_cmd_sub = self.create_subscription(
            Float64, "lift_cmd", self._lift_cmd_callback, 10
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.write_timer = self.create_timer(self.send_interval, self._send_control_frame)
        self.read_timer = self.create_timer(0.01, self._read_feedback_frame)

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self.cmd_vx = msg.linear.x
        self.cmd_vy = msg.linear.y
        self.cmd_wz = msg.angular.z

    def _lift_cmd_callback(self, msg: Float64) -> None:
        self.cmd_lift_vel = msg.data

    def _send_control_frame(self) -> None:
        r = self.wheel_radius
        l_sum = self.lx + self.ly

        w_a = (self.cmd_vx - self.cmd_vy - l_sum * self.cmd_wz) / r
        w_b = (self.cmd_vx + self.cmd_vy - l_sum * self.cmd_wz) / r
        w_c = (self.cmd_vx + self.cmd_vy + l_sum * self.cmd_wz) / r
        w_d = (self.cmd_vx - self.cmd_vy + l_sum * self.cmd_wz) / r

        w_lift = (self.cmd_lift_vel / self.lift_lead) * (2.0 * math.pi)

        rps_a = w_a / (2.0 * math.pi)
        rps_b = w_b / (2.0 * math.pi)
        rps_c = w_c / (2.0 * math.pi)
        rps_d = w_d / (2.0 * math.pi)
        rps_lift = w_lift / (2.0 * math.pi)

        dir_mask = 0x00

        if rps_a < 0:
            dir_mask |= 0x01
        val_a = min(65535, int(abs(rps_a) * 10.0))

        if rps_b < 0:
            dir_mask |= 0x02
        val_b = min(65535, int(abs(rps_b) * 10.0))

        if rps_c < 0:
            dir_mask |= 0x04
        val_c = min(65535, int(abs(rps_c) * 10.0))

        if rps_d < 0:
            dir_mask |= 0x08
        val_d = min(65535, int(abs(rps_d) * 10.0))

        if rps_lift < 0:
            dir_mask |= 0x10
        val_lift = min(65535, int(abs(rps_lift) * 10.0))

        frame = bytearray(15)
        frame[0] = 0xA5
        frame[1] = 0x5A
        frame[2] = 0x00
        frame[3] = dir_mask

        struct.pack_into(">H", frame, 4, val_a)
        struct.pack_into(">H", frame, 6, val_b)
        struct.pack_into(">H", frame, 8, val_c)
        struct.pack_into(">H", frame, 10, val_d)
        struct.pack_into(">H", frame, 12, val_lift)

        if self.debug_mode:
            frame[14] = 0x00
        else:
            checksum = sum(frame[:14]) & 0xFF
            frame[14] = checksum

        if self.ser and self.ser.is_open:
            try:
                self.ser.write(frame)
                self.ser.flush()
            except Exception as e:
                self.get_logger().error(f"Serial write error: {e}")
        else:
            if self.get_parameter("debug_mode").value:
                self.get_logger().debug(f"Mock serial TX: {frame.hex()}")

    def _read_feedback_frame(self) -> None:
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        feedback_received = False

        if self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting >= 15:
                    header = self.ser.read(2)
                    if header == b"\xa5\x5a" or header == b"\x5a\xa5":
                        payload = self.ser.read(13)

                        dir_mask = payload[1]
                        speeds = struct.unpack(">hhhhh", payload[2:12])

                        w_fb = []
                        for i in range(4):
                            spd = speeds[i] * 0.1 * (2.0 * math.pi)
                            if dir_mask & (1 << i):
                                spd = -spd
                            w_fb.append(spd)

                        spd_lift = speeds[4] * 0.1 * self.lift_lead
                        if dir_mask & 0x10:
                            spd_lift = -spd_lift

                        r = self.wheel_radius
                        l_sum = self.lx + self.ly
                        vx = (w_fb[0] + w_fb[1] + w_fb[2] + w_fb[3]) * r / 4.0
                        vy = (-w_fb[0] + w_fb[1] + w_fb[2] - w_fb[3]) * r / 4.0
                        wz = (-w_fb[0] - w_fb[1] + w_fb[2] + w_fb[3]) * r / (4.0 * l_sum)

                        self.lift_height += spd_lift * dt
                        feedback_received = True
            except Exception as e:
                self.get_logger().error(f"Serial read error: {e}")

        if not feedback_received:
            vx = self.cmd_vx
            vy = self.cmd_vy
            wz = self.cmd_wz
            self.lift_height += self.cmd_lift_vel * dt

        delta_x = (vx * math.cos(self.th) - vy * math.sin(self.th)) * dt
        delta_y = (vx * math.sin(self.th) + vy * math.cos(self.th)) * dt
        delta_th = wz * dt

        self.x += delta_x
        self.y += delta_y
        self.th += delta_th

        self.lift_height = max(0.0, min(0.5, self.lift_height))

        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        q = self._euler_to_quaternion(0.0, 0.0, self.th)
        odom.pose.pose.orientation = q

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = q
        self.tf_broadcaster.sendTransform(t)

        joint_state = JointState()
        joint_state.header.stamp = current_time.to_msg()
        joint_state.name = ["lift_joint"]
        joint_state.position = [self.lift_height]
        joint_state.velocity = [self.cmd_lift_vel]
        self.joint_pub.publish(joint_state)

        self.last_time = current_time

    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q

    def destroy_node(self) -> None:
        if self.ser and self.ser.is_open:
            stop_frame = bytearray(15)
            stop_frame[0] = 0xA5
            stop_frame[1] = 0x5A
            self.ser.write(stop_frame)
            self.ser.close()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", default="/dev/ttyUSB_chassis", help="Chassis serial port device path"
    )
    parser.add_argument(
        "--baudrate", type=int, default=115200, help="Baud rate for serial communication"
    )
    parsed_args, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = ChassisSerialBridge(port=parsed_args.port, baudrate=parsed_args.baudrate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

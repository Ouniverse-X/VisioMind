import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchSubstitution, LaunchConfiguration

def generate_launch_description():
    # Declare port/device parameters
    head_cam_device_arg = DeclareLaunchArgument(
        'head_cam_device',
        default_value='/dev/video_head',
        description='Device path for head USB camera (0c45:636b)'
    )
    left_arm_cam_device_arg = DeclareLaunchArgument(
        'left_arm_cam_device',
        default_value='/dev/video_left_arm',
        description='Device path for left arm USB camera (0c45:636b)'
    )
    right_arm_cam_device_arg = DeclareLaunchArgument(
        'right_arm_cam_device',
        default_value='/dev/video_right_arm',
        description='Device path for right arm USB camera (0c45:636b)'
    )

    # 1. Head USB Camera Node
    head_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='head_camera_node',
        namespace='head_camera',
        parameters=[{
            'video_device': LaunchConfiguration('head_cam_device'),
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'yuyv',
            'camera_frame_id': 'head_camera_optical_frame',
            'io_method': 'mmap',
            'frame_id': 'head_camera_link',
            'brightness': 128,
            'contrast': 128,
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('camera_info', 'camera_info'),
        ],
        output='screen'
    )

    # 2. Left Arm USB Camera Node
    left_arm_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='left_arm_camera_node',
        namespace='left_arm_camera',
        parameters=[{
            'video_device': LaunchConfiguration('left_arm_cam_device'),
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'yuyv',
            'camera_frame_id': 'left_arm_camera_optical_frame',
            'io_method': 'mmap',
            'frame_id': 'left_arm_camera_link',
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('camera_info', 'camera_info'),
        ],
        output='screen'
    )

    # 3. Right Arm USB Camera Node
    right_arm_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='right_arm_camera_node',
        namespace='right_arm_camera',
        parameters=[{
            'video_device': LaunchConfiguration('right_arm_cam_device'),
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'yuyv',
            'camera_frame_id': 'right_arm_camera_optical_frame',
            'io_method': 'mmap',
            'frame_id': 'right_arm_camera_link',
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('camera_info', 'camera_info'),
        ],
        output='screen'
    )

    # 4. Chest Depth Camera (Linux Foundation Aurora 930 - ID 3251:1930)
    # The depth camera is fully compatible with standard ROS 2 depth sensor pipelines
    # (such as realsense2_camera or specialized driver node).
    chest_depth_cam_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='chest_depth_camera_node',
        namespace='camera',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'align_depth.enable': True,  # Generate depth aligned to color frame
            'pointcloud.enable': True,   # Publish color pointcloud
            'base_frame_id': 'chest_camera_link',
            'depth_frame_id': 'chest_camera_depth_frame',
            'color_frame_id': 'chest_camera_color_frame',
        }],
        remappings=[
            ('depth/image_rect_raw', 'depth/image_rect_raw'),
            ('color/image_raw', 'color/image_raw'),
            ('depth/color/points', 'depth/color/points'),
        ],
        output='screen'
    )

    # 5. Static TF publishers mapping sensor links to robot footprint
    # Usually provided by robot state publisher, but here static publishers
    # are included as reference and fallback interfaces.
    tf_head_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.1', '0.0', '1.65', '0.0', '0.45', '0.0', 'base_footprint', 'head_camera_link']
    )
    tf_chest_cam = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.18', '0.0', '1.25', '0.0', '0.0', '0.0', 'base_footprint', 'chest_camera_link']
    )

    return LaunchDescription([
        head_cam_device_arg,
        left_arm_cam_device_arg,
        right_arm_cam_device_arg,
        head_cam_node,
        left_arm_cam_node,
        right_arm_cam_node,
        chest_depth_cam_node,
        tf_head_cam,
        tf_chest_cam
    ])

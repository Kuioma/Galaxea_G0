# Galaxea Open-World Dataset

## RLDS Format

#### RLDS Dataset Schema

```
RLDSDataset = {
    "episode_metadata": {
        "file_path": tf.Text,  # path to the original data file
    },
    "steps": {
        "is_first": tf.Scalar(dtype=bool),  # true on first step of the episode
        "is_last": tf.Scalar(dtype=bool),  # true on last step of the episode

        "language_instruction": tf.Text,  # language instruction, format: "high level"@"low level chinese"@"low level english"
        "observation": {
            "base_velocity": tf.Tensor(3, dtype=float32),   # robot base velocity
            "gripper_state_left": tf.Tensor(1, dtype=float32),  # left gripper state, 0-close and 100-open
            "gripper_state_right": tf.Tensor(1, dtype=float32), # right gripper state, 0-close and 100-open
            "depth_camera_wrist_left": tf.Tensor(224, 224, 1, dtype=uint16),  # wrist camera depth left viewpoint, unit: mm
            "depth_camera_wrist_right": tf.Tensor(224, 224, 1, dtype=uint16),  # wrist camera depth right viewpoint, unit: mm
            "image_camera_head": tf.Tensor(224, 224, 3, dtype=uint8), # head camera RGB viewpoint
            "image_camera_wrist_left": tf.Tensor(224, 224, 3, dtype=uint8), # wrist camera RGB left viewpoint
            "image_camera_wrist_right": tf.Tensor(224, 224, 3, dtype=uint8), # wrist camera RGB right viewpoint
            "joint_position_arm_left": tf.Tensor(6, dtype=float32), # joint positions of the left arm
            "joint_position_arm_right": tf.Tensor(6, dtype=float32), # joint positions of the right arm
            "joint_position_torso": tf.Tensor(4, dtype=float32), # joint positions of the torso
            "joint_velocity_arm_left": tf.Tensor(6, dtype=float32), # joint velocities of the left arm
            "joint_velocity_arm_right": tf.Tensor(6, dtype=float32), # joint velocities of the right arm
            "last_action": tf.Tensor(26, dtype=float32), # history of the last action
        },
        # action dimensions:
        # 26 = 6 (left arm) + 1 (left gripper) + 6 (right arm) + 1 (right gripper) + 6 (torso) + 6 (base)
        "action": tf.Tensor(26, dtype=float32),  # robot action, consists of [6x joint velocities, 1x gripper position]
        "segment_idx": tf.Scalar(dtype=int32),  # index of the segment in the episode
        "variant_idx": tf.Scalar(dtype=int32), 
    },
}
```

#### Lerobot Dataset Schema in parquet
```
LerobotDataset = {
    "features": {
        "observation.images.head_rgb": {
            "dtype": "video",
            "shape": [
                720,
                1280,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 720,
                "video.width": 1280,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 15,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.head_right_rgb": {
            "dtype": "video",
            "shape": [
                720,
                1280,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 720,
                "video.width": 1280,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 15,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.left_wrist_rgb": {
            "dtype": "video",
            "shape": [
                720,
                1280,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 720,
                "video.width": 1280,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 15,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.right_wrist_rgb": {
            "dtype": "video",
            "shape": [
                720,
                1280,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 720,
                "video.width": 1280,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 15,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.state.left_arm": {
            "dtype": "float64",
            "shape": [
                7
            ],
            "names": [
                "/hdas/feedback_arm_left.position[0]",
                "/hdas/feedback_arm_left.position[1]",
                "/hdas/feedback_arm_left.position[2]",
                "/hdas/feedback_arm_left.position[3]",
                "/hdas/feedback_arm_left.position[4]",
                "/hdas/feedback_arm_left.position[5]",
                "/hdas/feedback_arm_left.position[6]"
            ]
        },
        "observation.state.left_arm.velocities": {
            "dtype": "float64",
            "shape": [
                7
            ],
            "names": [
                "/hdas/feedback_arm_left.velocity[0]",
                "/hdas/feedback_arm_left.velocity[1]",
                "/hdas/feedback_arm_left.velocity[2]",
                "/hdas/feedback_arm_left.velocity[3]",
                "/hdas/feedback_arm_left.velocity[4]",
                "/hdas/feedback_arm_left.velocity[5]",
                "/hdas/feedback_arm_left.velocity[6]"
            ]
        },
        "observation.state.right_arm": {
            "dtype": "float64",
            "shape": [
                7
            ],
            "names": [
                "/hdas/feedback_arm_right.position[0]",
                "/hdas/feedback_arm_right.position[1]",
                "/hdas/feedback_arm_right.position[2]",
                "/hdas/feedback_arm_right.position[3]",
                "/hdas/feedback_arm_right.position[4]",
                "/hdas/feedback_arm_right.position[5]",
                "/hdas/feedback_arm_right.position[6]"
            ]
        },
        "observation.state.right_arm.velocities": {
            "dtype": "float64",
            "shape": [
                7
            ],
            "names": [
                "/hdas/feedback_arm_right.velocity[0]",
                "/hdas/feedback_arm_right.velocity[1]",
                "/hdas/feedback_arm_right.velocity[2]",
                "/hdas/feedback_arm_right.velocity[3]",
                "/hdas/feedback_arm_right.velocity[4]",
                "/hdas/feedback_arm_right.velocity[5]",
                "/hdas/feedback_arm_right.velocity[6]"
            ]
        },
        "observation.state.chassis": {
            "dtype": "float64",
            "shape": [
                10
            ],
            "names": [
                "/hdas/imu_chassis.orientation.x",
                "/hdas/imu_chassis.orientation.y",
                "/hdas/imu_chassis.orientation.z",
                "/hdas/imu_chassis.orientation.w",
                "/hdas/imu_chassis.angular_velocity.x",
                "/hdas/imu_chassis.angular_velocity.y",
                "/hdas/imu_chassis.angular_velocity.z",
                "/hdas/imu_chassis.linear_acceleration.x",
                "/hdas/imu_chassis.linear_acceleration.y",
                "/hdas/imu_chassis.linear_acceleration.z"
            ]
        },
        "observation.state.torso": {
            "dtype": "float64",
            "shape": [
                4
            ],
            "names": [
                "/hdas/feedback_torso.position[0]",
                "/hdas/feedback_torso.position[1]",
                "/hdas/feedback_torso.position[2]",
                "/hdas/feedback_torso.position[3]"
            ]
        },
        "observation.state.torso.velocities": {
            "dtype": "float64",
            "shape": [
                4
            ],
            "names": [
                "/hdas/feedback_torso.velocity[0]",
                "/hdas/feedback_torso.velocity[1]",
                "/hdas/feedback_torso.velocity[2]",
                "/hdas/feedback_torso.velocity[3]"
            ]
        },
        "observation.state.left_gripper": {
            "dtype": "float64",
            "shape": [
                1
            ],
            "names": [
                "/hdas/feedback_gripper_left.position[0]"
            ]
        },
        "observation.state.right_gripper": {
            "dtype": "float64",
            "shape": [
                1
            ],
            "names": [
                "/hdas/feedback_gripper_right.position[0]"
            ]
        },
        "observation.state.left_ee_pose": {
            "dtype": "float64",
            "shape": [
                7
            ],
            "names": [
                "/motion_control/pose_ee_arm_left.pose.position.x",
                "/motion_control/pose_ee_arm_left.pose.position.y",
                "/motion_control/pose_ee_arm_left.pose.position.z",
                "/motion_control/pose_ee_arm_left.pose.orientation.x",
                "/motion_control/pose_ee_arm_left.pose.orientation.y",
                "/motion_control/pose_ee_arm_left.pose.orientation.z",
                "/motion_control/pose_ee_arm_left.pose.orientation.w"
            ]
        },
        "observation.state.right_ee_pose": {
            "dtype": "float64",
            "shape": [
                7
            ],
            "names": [
                "/motion_control/pose_ee_arm_right.pose.position.x",
                "/motion_control/pose_ee_arm_right.pose.position.y",
                "/motion_control/pose_ee_arm_right.pose.position.z",
                "/motion_control/pose_ee_arm_right.pose.orientation.x",
                "/motion_control/pose_ee_arm_right.pose.orientation.y",
                "/motion_control/pose_ee_arm_right.pose.orientation.z",
                "/motion_control/pose_ee_arm_right.pose.orientation.w"
            ]
        },
        "action.left_gripper": {
            "dtype": "float64",
            "shape": [
                1
            ],
            "names": [
                "/motion_target/target_position_gripper_left.position[0]"
            ]
        },
        "action.right_gripper": {
            "dtype": "float64",
            "shape": [
                1
            ],
            "names": [
                "/motion_target/target_position_gripper_right.position[0]"
            ]
        },
        "action.chassis.velocities": {
            "dtype": "float64",
            "shape": [
                3
            ],
            "names": [
                "/motion_target/target_speed_chassis.twist.linear.x",
                "/motion_target/target_speed_chassis.twist.linear.y",
                "/motion_target/target_speed_chassis.twist.angular.z"
            ]
        },
        "action.left_arm": {
            "dtype": "float64",
            "shape": [
                6
            ],
            "names": [
                "/motion_target/target_joint_state_arm_left.position[0]",
                "/motion_target/target_joint_state_arm_left.position[1]",
                "/motion_target/target_joint_state_arm_left.position[2]",
                "/motion_target/target_joint_state_arm_left.position[3]",
                "/motion_target/target_joint_state_arm_left.position[4]",
                "/motion_target/target_joint_state_arm_left.position[5]"
            ]
        },
        "action.right_arm": {
            "dtype": "float64",
            "shape": [
                6
            ],
            "names": [
                "/motion_target/target_joint_state_arm_right.position[0]",
                "/motion_target/target_joint_state_arm_right.position[1]",
                "/motion_target/target_joint_state_arm_right.position[2]",
                "/motion_target/target_joint_state_arm_right.position[3]",
                "/motion_target/target_joint_state_arm_right.position[4]",
                "/motion_target/target_joint_state_arm_right.position[5]"
            ]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": { # step index in the episode
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "coarse_task_index": { # high level instruction index in tasks.jsonl
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": { # index for the low level instruction in episodes.jsonl
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "coarse_quality_index": { # episode quality label(None or unqualified) index in tasks.jsonl
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "quality_index": { # atomtic quality label(qualified or unqualified) index in tasks.jsonl
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```
#### Example

We provide an example script to load our RLDS dataset and transform some episodes into mp4 video format (head camera).

```python
import tensorflow_datasets as tfds
import tyro
import os
import imageio
from tqdm import tqdm

def main(
    dataset_name: str, 
    data_dir: str, 
    output_dir: str = "extracted_videos",
    num_trajs: int = 10
):
    ds = tfds.load(dataset_name, split='train', data_dir=data_dir)
    print(f"Successfully loaded dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Videos will be saved to: {output_dir}")

    for i, episode in enumerate(tqdm(ds.take(num_trajs), total=num_trajs, desc="Exporting videos")):
        head_frames = []
        
        for step in episode['steps']:
            head_rgb_image = step['observation']['image_camera_head'].numpy()
            head_frames.append(head_rgb_image)
            instruction = step['language_instruction'].numpy().decode('utf-8')

        video_path = os.path.join(output_dir, f"traj_{i}_head_rgb.mp4")
        try:
            imageio.mimsave(video_path, head_frames, fps=15)
            print(f"Saved video for episode {i} to {video_path} with instruction: '{instruction}'")
        except Exception as e:
            print(f"Error saving video for episode {i}: {e}")

if __name__ == '__main__':
    tyro.cli(main)
```



## Lerobot Format

Coming Soon !

import math
import torch
from tensordict import TensorDict
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from go2_curriculum import CommandCurriculum


# 辅助函数：生成指定范围内的随机数张量
# lower: 下界, upper: 上界, batch_shape: 批次形状（比如多少个环境）
def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    # 公式：(上界 - 下界) * 随机数 + 下界
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


class Go2Env:
    # 初始化函数：创建环境时自动运行
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs: int = num_envs  # 并行环境的数量（比如4096个）
        self.num_actions = env_cfg["num_actions"]  # 动作数量（12个关节）
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]  # 指令数量（3个：x速度, y速度, 角速度）
        self.device = gs.device  # 计算设备（CPU或GPU）

        self.simulate_action_latency = True  # 模拟真实机器人的动作延迟（1步延迟）
        self.dt = 0.02  # 时间步长（0.02秒，即50Hz控制频率）
        # 计算一个episode（回合）最大有多少步
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # 保存各种配置字典
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        # 提取观测缩放比例和奖励缩放比例
        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        # --- 1. 创建物理场景 ---
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,  # 仿真步长
                substeps=2,  # 子步数，提高仿真精度
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,  # 关闭自碰撞（机器狗身体部件之间不碰撞，省算力）
                max_collision_pairs=20,  # 最大碰撞对数量
                tolerance=1e-5,  # 求解器精度
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],  # 只渲染第0个环境（为了看画面，其他环境在后台跑）
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),  # 相机位置
                camera_lookat=(0.0, 0.0, 0.5),  # 相机看向哪里
                camera_fov=40,  # 视野角度
            ),
            show_viewer=show_viewer,  # 是否显示窗口
        )

        # 添加地面（一个固定的URDF平面）
        self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # 添加机器狗（加载URDF模型，设置初始位置和姿态）
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],  # 初始位置 [0, 0, 0.42]
                quat=self.env_cfg["base_init_quat"],  # 初始姿态 [1, 0, 0, 0] (四元数)
            ),
        )

        # 构建场景（必须在添加完实体后调用，n_envs指定并行数量）
        self.scene.build(n_envs=num_envs)

        # --- 2. 获取关节索引 ---
        # 获取12个电机对应的自由度索引
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        # 对索引进行排序（为了控制顺序一致）
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        # --- 3. 设置 PD 控制参数 ---
        # kp: 比例系数（刚度），kd: 微分系数（阻尼）
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # --- 4. 定义全局变量和初始状态 ---
        # 全局重力方向向量 [0, 0, -1]
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # 初始机身位置和姿态
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)  # 初始姿态的逆四元数（用于坐标变换）

        # 初始关节角度（从配置中读取）
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],  # joints[0]通常是浮空基座
            dtype=gs.tc_float,
            device=gs.device,
        )
        # 拼接成完整的初始状态（位置+姿态+关节角）
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        # 计算初始状态下的投影重力（用于观测）
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # --- 5. 初始化缓冲区（Buffers） ---
        # 这些是用来存储每一步数据的“表格”，大小都是 (num_envs, ...)
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)  # 线速度
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)  # 角速度
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)  # 投影重力
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)  # 奖励缓冲区
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)  # 重置标记（初始全为True）
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)  # 回合长度计数
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)  # 指令（目标速度）

        # 指令缩放比例（用于观测归一化）
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        # 指令的上下限（从配置读取，比如x速度范围[0.5, 0.5]）
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        # 旧实验没有 curriculum 配置时，仍按原来的固定范围采样。
        curriculum_cfg = self.command_cfg.get("curriculum")
        self.command_curriculum = (
            CommandCurriculum(self.commands_limits, curriculum_cfg, self.num_envs, self.dt)
            if curriculum_cfg is not None else None
        )

        # 动作相关缓冲区
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)  # 上一步的动作
        self.dof_pos = torch.empty_like(self.actions)  # 当前关节位置
        self.dof_vel = torch.empty_like(self.actions)  # 当前关节速度
        self.last_dof_vel = torch.zeros_like(self.actions)  # 上一步关节速度

        # 机身状态缓冲区
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)  # 欧拉角（roll, pitch, yaw）

        # 默认关节位置（用于计算偏差）
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.extras = dict()  # 用于记录额外信息（如日志）

        # --- 6. 特殊奖励函数的初始化 ---
        # 如果配置了“关节目标限制”奖励
        if "joint_target_limits" in self.reward_scales:
            lower, upper = self.robot.get_dofs_limit(self.motors_dof_idx)  # 获取关节物理限位
            margin_ratio = self.reward_cfg.get("joint_target_limit_margin", 0.05)  # 缓冲比例
            if not 0.0 <= margin_ratio < 0.5:
                raise ValueError("joint_target_limit_margin must be in [0, 0.5).")
            margin = (upper - lower) * margin_ratio  # 计算缓冲区域大小
            self.joint_target_lower = lower + margin  # 缩小后的下限
            self.joint_target_upper = upper - margin  # 缩小后的上限

        # 脚部使用率和对角支撑奖励共用真实的足端接触数据。
        if "feet_usage" in self.reward_scales or "diagonal_support" in self.reward_scales:
            feet_geom_indices = []
            # 寻找四个脚掌的碰撞球体索引
            for leg in ("FL", "FR", "RL", "RR"):
                candidates = [
                    geom for geom in self.robot.geoms
                    if geom.link.name in (f"{leg}_calf", f"{leg}_foot")  # 小腿或脚掌
                       and geom.type == gs.GEOM_TYPE.SPHERE  # 必须是球体
                       and abs(float(geom.data[0]) - 0.022) < 1e-5  # 半径约为0.022
                ]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one Go2 foot collision sphere for {leg}, found {len(candidates)}.")
                feet_geom_indices.append(candidates[0].idx)
            self.feet_geom_indices = torch.tensor(feet_geom_indices, dtype=gs.tc_int, device=gs.device)
            self.feet_contact_force_threshold = self.reward_cfg.get("feet_contact_force_threshold", 5.0)  # 接触力阈值
            # 顺序固定为：左前 FL、右前 FR、左后 RL、右后 RR；1 表示支撑，0 表示未支撑。
            self.feet_contact = torch.zeros((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)

        if "feet_usage" in self.reward_scales:
            self.feet_usage_target = self.reward_cfg.get("feet_usage_target", 0.3)  # 目标使用率
            time_constant = self.reward_cfg.get("feet_usage_time_constant", 0.4)  # 时间常数（平滑用）
            self.feet_usage_min_command = self.reward_cfg.get("feet_usage_min_command", 0.2)  # 最小指令速度（静止时不惩罚）
            if not 0.0 < self.feet_usage_target <= 1.0 or time_constant <= 0.0:
                raise ValueError("feet_usage_target must be in (0, 1] and feet_usage_time_constant must be positive.")
            self.feet_contact_alpha = self.dt / (time_constant + self.dt)  # 平滑系数
            # 初始化脚部接触平均值（填0.35防止初始惩罚过大）
            self.feet_contact_average = torch.full((self.num_envs, 4), 0.35, dtype=gs.tc_float, device=gs.device)

        # --- 7. 准备奖励函数映射 ---
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt  # 奖励乘以时间步长（积分效果）
            # 动态获取类中名为 _reward_xxx 的方法
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            # 初始化回合奖励累加器
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # 执行一次重置，让所有机器狗回到初始状态
        self.reset()

    # 重新采样指令（给机器狗新的目标速度）
    def _resample_commands(self, envs_idx):
        # 在 limits 范围内随机生成指令
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)  # 全部重置
        else:
            # 只重置指定的环境（envs_idx为True的那些）
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    # 核心步进函数（强化学习每一步都会调用这个）
    def step(self, actions):
        # 1. 限制动作范围（防止输出过大）
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # 2. 处理动作延迟（如果是True，就用上一步的动作）
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # 3. 计算目标关节位置：默认位置 + 动作 * 缩放系数
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # 4. 发送控制指令给物理引擎（控制第6到18个自由度，即12个关节）
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))

        # 5. 物理引擎步进（世界向前走0.02秒）
        self.scene.step()

        # --- 更新缓冲区数据 ---
        self.episode_length_buf += 1  # 回合计数+1
        self.base_pos = self.robot.get_pos()  # 获取机身位置
        self.base_quat = self.robot.get_quat()  # 获取机身姿态（四元数）

        # 计算欧拉角（roll, pitch, yaw），用于判断是否摔倒
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )

        inv_base_quat = inv_quat(self.base_quat)  # 当前姿态的逆
        # 将世界坐标系的速度转换到机身坐标系
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        # 将重力向量转换到机身坐标系（这样机器狗就知道哪边是下）
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # 获取关节位置和速度
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # --- 计算奖励 ---
        if hasattr(self, "feet_contact"):
            self._update_feet_contact()  # 每个物理步只读取一次，供两项奖励共同使用。
        self.rew_buf.zero_()  # 清空奖励缓冲区
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]  # 计算单项奖励 * 权重
            self.rew_buf += rew  # 累加到总奖励
            self.episode_sums[name] += rew  # 累加到回合统计

        # --- 检查终止条件 ---
        self.reset_buf = self.episode_length_buf > self.max_episode_length  # 时间到了
        # 俯仰角（pitch）太大（前后翻）
        failed = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        # 翻滚角（roll）太大（左右翻）
        failed |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # 物理引擎报错（比如穿模了）
        failed |= self.scene.rigid_solver.get_error_envs_mask()
        self.reset_buf |= failed

        # 先用本步实际执行的指令评估课程，再重采样或重置，避免错配新旧指令。
        if self.command_curriculum is not None and self.command_curriculum.enabled:
            self.command_curriculum.update(self._reward_tracking_lin_vel(), self._reward_tracking_ang_vel(), failed)

        # 每隔一段时间重新采样指令（比如每4秒），升级后的范围在下次采样时生效。
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # 记录超时信息（用于区分是正常结束还是摔倒结束）
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # 如果需要重置，执行重置逻辑
        self._reset_idx(self.reset_buf)
        if self.command_curriculum is not None:
            self.extras["episode"].update(self.command_curriculum.metrics)

        # 更新观测值（给神经网络看的数据）
        self._update_observation()

        # 保存当前动作和速度，供下一步使用（用于计算动作平滑奖励等）
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        # 返回：观测值, 奖励, 是否重置, 额外信息
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        # 将观测数据打包成 TensorDict 返回
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # 重置函数（把摔倒或结束的机器狗恢复原状）
    def _reset_idx(self, envs_idx=None):
        # 1. 重置物理状态（位置、姿态、速度归零）
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # 2. 重置缓冲区数据
        if envs_idx is None:
            # 如果没指定索引，重置所有环境
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            # 如果指定了索引，只重置那些环境（使用 torch.where 进行条件赋值）
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # 重置脚部接触历史（如果有的话）
        if hasattr(self, "feet_contact"):
            if envs_idx is None:
                self.feet_contact.zero_()
            else:
                self.feet_contact.masked_fill_(envs_idx[:, None], 0.0)
        if hasattr(self, "feet_contact_average"):
            if envs_idx is None:
                self.feet_contact_average.fill_(0.35)
            else:
                self.feet_contact_average.masked_fill_(envs_idx[:, None], 0.35)

        # 3. 记录日志（计算这个回合的平均奖励）
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]  # 平均每秒奖励
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # 4. 重置时随机给一个新的指令
        self._resample_commands(envs_idx)

    # 更新观测值（把各种数据拼成一个长向量，喂给神经网络）
    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 角速度 (3维)
                self.projected_gravity,  # 投影重力 (3维) - 告诉狗哪边是下
                self.commands * self.commands_scale,  # 目标指令 (3维) - 告诉狗要去哪
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 关节位置偏差 (12维)
                self.dof_vel * self.obs_scales["dof_vel"],  # 关节速度 (12维)
                self.actions,  # 上一步动作 (12维)
            ),
            dim=-1,  # 在最后一个维度拼接，总共 3+3+3+12+12+12 = 45维
        )

    def reset(self):
        # 初始重置
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ================= 奖励函数区域 =================
    # 注意：函数名必须和 reward_cfg 里的 key 对应（去掉 _reward_ 前缀）

    def _reward_tracking_lin_vel(self):
        # 奖励：跟踪线速度（x, y方向）
        # 计算目标速度和实际速度的误差平方和
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 使用指数函数：误差越小，exp(0)=1，奖励越大；误差越大，奖励趋近0
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # 奖励：跟踪角速度（yaw，即转弯）
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # 惩罚：z轴（垂直方向）线速度
        # 机器狗走路应该平稳，不应该上下乱跳，所以这个值是负的（在cfg里配置了负权重）
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # 惩罚：动作变化率
        # 鼓励动作平滑，不要剧烈抖动（当前动作 - 上一步动作）
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # 惩罚：偏离默认姿态
        # 鼓励机器狗保持默认的站立姿势，不要扭曲
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # 惩罚：机身高度偏差
        # 鼓励机器狗保持目标高度（比如0.3米）
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_joint_target_limits(self):
        # 惩罚：关节目标角度越界
        # 计算目标位置（默认位置 + 动作*缩放）
        target = self.default_dof_pos + self.env_cfg["action_scale"] * self.actions
        # 计算超出安全范围（lower, upper）的部分
        excess = (self.joint_target_lower - target).clamp_min(0) + (target - self.joint_target_upper).clamp_min(0)
        return excess.square().sum(dim=-1)

    def _update_feet_contact(self):
        # 读取四个脚掌各自的真实接触力；身体或小腿碰地不算脚掌支撑。
        contacts = self.robot.get_contacts(is_padded=True)  # 获取接触信息
        valid = contacts["valid_mask"][:, :, None]
        # 检查哪些几何体是脚掌
        matches_a = (contacts["geom_a"][:, :, None] == self.feet_geom_indices) & valid
        matches_b = (contacts["geom_b"][:, :, None] == self.feet_geom_indices) & valid
        # 计算脚掌受到的垂直力（z轴）
        force_z = (
                matches_a * contacts["force_a"][:, :, 2, None]
                + matches_b * contacts["force_b"][:, :, 2, None]
        ).sum(dim=1)
        # 如果力大于阈值，认为接触了（1.0），否则没接触（0.0）
        self.feet_contact.copy_((force_z > self.feet_contact_force_threshold).to(gs.tc_float))

    def _reward_feet_usage(self):
        # 惩罚某条腿长期不参与支撑；这一项本身不规定迈腿顺序，也不能单独防止蹦跳。
        # 使用低通滤波平滑接触信号
        self.feet_contact_average.lerp_(self.feet_contact, self.feet_contact_alpha)
        # 计算使用率不足的部分（目标0.3，如果实际只有0.1，deficit就是0.2）
        deficit = ((self.feet_usage_target - self.feet_contact_average).clamp_min(0) / self.feet_usage_target)
        # 只有在机器狗被要求移动时（指令速度 > 0.2），才惩罚脚部使用率不足
        moving = torch.linalg.vector_norm(self.commands[:, :2], dim=-1) > self.feet_usage_min_command
        return deficit.square().sum(dim=-1) * moving

    def _reward_diagonal_support(self):
        # 奖励仅有一组对角脚支撑：左前+右后，或右前+左后。
        # 四脚同时着地、全腾空、两只前脚、两只后脚或同侧两脚支撑，都不获得此项奖励。
        fl, fr, rl, rr = self.feet_contact.unbind(dim=-1)
        diagonal_a = fl * rr * (1.0 - fr) * (1.0 - rl)
        diagonal_b = fr * rl * (1.0 - fl) * (1.0 - rr)
        min_command = self.reward_cfg.get("feet_usage_min_command", 0.2)
        moving = torch.linalg.vector_norm(self.commands[:, :2], dim=-1) > min_command
        # 不指定节拍；需要结合四脚使用率和实测时序确认两组确实交替。
        return (diagonal_a + diagonal_b) * moving

"""根据速度跟踪和稳定性，逐步扩大 Go2 的指令采样范围。"""

import math

import torch


class CommandCurriculum:
    def __init__(self, limits, cfg, num_envs, dt):
        self.limits = limits  # 与环境共用上下限张量，更新后重新采样即生效。
        self.enabled = cfg.get("enabled", False)
        self.num_levels = cfg.get("num_levels", 10)
        interval = cfg.get("update_interval_s", 20.0)
        self.linear_threshold = cfg.get("linear_tracking_threshold", 0.8)
        self.angular_threshold = cfg.get("angular_tracking_threshold", 0.9)
        self.survival_threshold = cfg.get("survival_threshold", 0.9)
        if not isinstance(self.num_levels, int) or self.num_levels < 1:
            raise ValueError("curriculum.num_levels must be a positive integer.")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("curriculum.update_interval_s must be finite and positive.")
        if not all(0 < x <= 1 for x in (self.linear_threshold, self.angular_threshold, self.survival_threshold)):
            raise ValueError("Curriculum thresholds must be in (0, 1].")
        self.interval_steps = max(1, math.ceil(interval / dt))
        self.initial = torch.stack(limits).clone()
        self.final = self.initial.new_tensor(
            [cfg["final_lin_vel_x_range"], cfg["final_lin_vel_y_range"], cfg["final_ang_vel_range"]]
        ).T
        if self.initial.shape != (2, 3) or self.final.shape != (2, 3):
            raise ValueError("Command curriculum requires three [lower, upper] ranges.")
        if not torch.isfinite(self.initial).all() or not torch.isfinite(self.final).all():
            raise ValueError("Command ranges must be finite.")
        if not torch.all(self.initial[0] <= self.initial[1]):
            raise ValueError("Initial command lower bounds must not exceed upper bounds.")
        if not torch.all((self.final[0] <= self.initial[0]) & (self.final[1] >= self.initial[1])):
            raise ValueError("Final command ranges must contain the initial ranges.")
        self.score_sum = self.initial.new_zeros(2)
        self.failed = torch.zeros(num_envs, dtype=torch.bool, device=self.initial.device)
        self.metrics = {}
        self.set_level(0)

    def set_level(self, level):
        if not isinstance(level, int) or not 0 <= level <= self.num_levels:
            raise ValueError(f"Curriculum level must be an integer in [0, {self.num_levels}].")
        self.level = level
        # 第 0 级用初始范围，第 num_levels 级准确到达最终范围。
        progress = level / self.num_levels
        bounds = self.initial + progress * (self.final - self.initial)
        if level == self.num_levels:
            bounds = self.final
        self.limits[0].copy_(bounds[0])
        self.limits[1].copy_(bounds[1])
        self.metrics = {"Curriculum/level": level, "Curriculum/progress": progress}
        for axis, (lower, upper) in zip(("vx", "vy", "yaw"), bounds.T.tolist()):
            self.metrics[f"Curriculum/{axis}_min"] = lower
            self.metrics[f"Curriculum/{axis}_max"] = upper
        self._clear_window()

    def _clear_window(self):
        self.steps = 0
        self.score_sum.zero_()
        self.failed.zero_()

    def update(self, linear_score, angular_score, failed):
        if not self.enabled:
            return
        # 用实际控制步统计；不依赖训练器随机初始化的 episode_length_buf。
        self.score_sum += torch.stack((linear_score.mean(), angular_score.mean()))
        self.failed |= failed  # 一个环境在本窗口内失败过一次，就不算存活。
        self.steps += 1
        if self.steps < self.interval_steps:
            return
        linear, angular = (self.score_sum / self.steps).tolist()
        survival = (~self.failed).float().mean().item()
        if (
            self.level < self.num_levels
            # 容纳 float32 的舍入误差，例如 9/10 可能存成 0.899999976。
            and linear + 1e-6 >= self.linear_threshold
            and angular + 1e-6 >= self.angular_threshold
            and survival + 1e-6 >= self.survival_threshold
        ):
            self.set_level(self.level + 1)
        # 未达标时只重开统计窗口，保持当前难度；不会强制按训练轮数升级。
        self.metrics.update(
            {
                "Curriculum/linear_tracking": linear,
                "Curriculum/angular_tracking": angular,
                "Curriculum/survival": survival,
            }
        )
        self._clear_window()

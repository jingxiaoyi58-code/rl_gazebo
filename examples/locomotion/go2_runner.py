"""PPO 保存和加载模型时，同时保存和恢复 Go2 的课程等级。"""

from rsl_rl.runners import OnPolicyRunner


def entropy_coef_at_iteration(schedule, iteration):
    """按绝对轮号计算熵系数，续训时不会重新从高探索强度开始。"""
    start, end = schedule["start_iteration"], schedule["end_iteration"]
    progress = min(1.0, max(0.0, (iteration - start) / max(1, end - start)))
    if end <= start:
        progress = 1.0
    return schedule["initial_coef"] + progress * (schedule["final_coef"] - schedule["initial_coef"])


class Go2Runner(OnPolicyRunner):
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        schedule = self.cfg.get("entropy_schedule")
        if schedule is None:
            return super().learn(num_learning_iterations, init_at_random_ep_len)

        original_update = self.alg.update
        next_iteration = self.current_learning_iteration

        def update_with_entropy_schedule():
            nonlocal next_iteration
            # 在 PPO 更新前设置熵项权重；动作仍由 PPO 自己的分布采样和计算概率。
            self.alg.entropy_coef = entropy_coef_at_iteration(schedule, next_iteration)
            losses = original_update()
            losses["entropy_coef"] = self.alg.entropy_coef  # TensorBoard: Loss/entropy_coef
            next_iteration += 1
            return losses

        self.alg.update = update_with_entropy_schedule
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.update = original_update

    def save(self, path, infos=None):
        if "entropy_schedule" in self.cfg:
            infos = dict(infos or {})
            infos["exploration"] = {"entropy_coef": self.alg.entropy_coef,
                                    "entropy_schedule": dict(self.cfg["entropy_schedule"])}
        curriculum = self.env.command_curriculum
        if curriculum is not None:
            infos = dict(infos or {})
            infos["command_curriculum"] = {"level": curriculum.level}
        super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        curriculum = self.env.command_curriculum
        if curriculum is not None:
            level = (infos or {}).get("command_curriculum", {}).get("level", 0)
            curriculum.set_level(level)
            # 恢复后立刻使用对应范围，不能继续沿用构建环境时采样的初始指令。
            self.env._resample_commands(None)
            self.env._update_observation()
        return infos

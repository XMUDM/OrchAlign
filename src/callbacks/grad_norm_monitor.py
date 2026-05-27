import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

class GradNormMonitor(Callback):
    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval  # 控制记录频率

    @rank_zero_only  # 确保只在主进程记录
    def on_after_backward(self, trainer, pl_module):
        """安全计算并记录梯度范数"""
        try:
            # 确保 logger 和 experiment 存在
            if not hasattr(trainer, "logger") or trainer.logger is None:
                return
            
            # 确保当前步需要记录（减少开销）
            if trainer.global_step % self.log_interval != 0:
                return

            total_norm = 0.0
            grad_norms = {}
            has_valid_grad = False

            # 遍历所有参数
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    # 计算 L2 范数
                    param_norm = param.grad.detach().data.norm(2).item()
                    total_norm += param_norm ** 2
                    grad_norms[f"grad_norm/{name}"] = param_norm
                    has_valid_grad = True

            if not has_valid_grad:
                return  # 无有效梯度时跳过

            # 计算总范数
            total_norm = total_norm ** 0.5
            grad_norms["grad_norm/total"] = total_norm

            # 记录到 WandB
            if hasattr(trainer.logger, "experiment"):
                trainer.logger.experiment.log(
                    grad_norms,
                    step=trainer.global_step
                )
        except Exception as e:
            # 打印错误但避免中断训练
            import traceback
            traceback.print_exc()
            pl_module.print(f"梯度范数记录失败: {str(e)}")
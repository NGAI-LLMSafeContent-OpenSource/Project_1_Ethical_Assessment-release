
"""不同的距离/散度度量函数，用于计算基础模型和有毒模型输出之间的差异。

包括：KL散度、Jensen-Shannon散度、总变分距离和Wasserstein距离等度量方法。
"""

import enum

import torch
import geomloss
import transformers


def fkl_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """计算正向KL散度 (Forward KL Divergence)。
    
    衡量有毒模型分布相对于基础模型分布的差异。对有毒模型给高概率但基础模型
    给低概率的情况惩罚较重。
    
    Args:
        base_logits: 基础模型的logits输出，shape: (batch_size, vocab_size)
        toxic_logits: 有毒模型的logits输出，shape: (batch_size, vocab_size)
        **kwargs: 其他参数（用于与其他函数接口统一）
    
    Returns:
        torch.Tensor: 正向KL散度值（标量）
    """
    fkl_score = torch.nn.functional.kl_div(
        input=toxic_logits.log_softmax(dim=-1),
        target=base_logits.softmax(dim=-1),
        reduction='batchmean',
        log_target=False
    )
    return fkl_score


def rkl_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """计算反向KL散度 (Reverse KL Divergence)。
    
    衡量基础模型分布相对于有毒模型分布的差异。对基础模型给高概率但有毒模型
    给低概率的情况惩罚较重，倾向于保留基础模型的关键特征。
    
    Args:
        base_logits: 基础模型的logits输出，shape: (batch_size, vocab_size)
        toxic_logits: 有毒模型的logits输出，shape: (batch_size, vocab_size)
        **kwargs: 其他参数（用于与其他函数接口统一）
    
    Returns:
        torch.Tensor: 反向KL散度值（标量）
    """
    rkl_score = torch.nn.functional.kl_div(
        input=base_logits.log_softmax(dim=-1),
        target=toxic_logits.softmax(dim=-1),
        reduction='batchmean',
        log_target=False
    )
    return rkl_score


def js_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """计算Jensen-Shannon散度 (JS Divergence)。
    
    对称的KL散度度量，是正向KL和反向KL的加权平均。相比单独的KL散度，
    提供了更对称和稳定的距离度量。
    
    JS = 0.5 * KL(base || toxic) + 0.5 * KL(toxic || base)
    
    Args:
        base_logits: 基础模型的logits输出，shape: (batch_size, vocab_size)
        toxic_logits: 有毒模型的logits输出，shape: (batch_size, vocab_size)
        **kwargs: 其他参数（用于与其他函数接口统一）
    
    Returns:
        torch.Tensor: Jensen-Shannon散度值（标量）
    """
    js_score = 0.5 * fkl_diffs_fn(base_logits, toxic_logits) + 0.5 * rkl_diffs_fn(base_logits, toxic_logits)
    
    return js_score


def tvd_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """计算总变分距离 (Total Variation Distance, TVD)。
    
    衡量两个概率分布之间的最大差异，对所有维度的绝对差值求和的一半。
    TVD的范围在0到1之间，0表示分布完全相同，1表示完全不同。
    
    TVD(p, q) = 0.5 * sum(|p_i - q_i|)
    
    Args:
        base_logits: 基础模型的logits输出，shape: (batch_size, vocab_size)
        toxic_logits: 有毒模型的logits输出，shape: (batch_size, vocab_size)
        **kwargs: 其他参数（用于与其他函数接口统一）
    
    Returns:
        torch.Tensor: 总变分距离值（标量）
    """
    p = torch.nn.functional.softmax(base_logits, dim=-1)
    q = torch.nn.functional.softmax(toxic_logits, dim=-1)

    tvd_score = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
    
    return tvd_score


def wsd_diffs_fn(base_logits: torch.Tensor, toxic_logits: torch.Tensor, wsd_p: int = 1, wsd_blur: float = 0.01) -> torch.Tensor:
    """计算Wasserstein切片距离 (Wasserstein Sliced Distance, WSD)。n    
    使用最优传输理论计算两个分布间的距离。考虑了类别间的相似性结构，
    因此相比KL散度等度量方法更具有几何意义。基于Sinkhorn算法实现。
    
    Args:
        base_logits: 基础模型的logits输出，shape: (batch_size, vocab_size) 或 (batch_size, seq_len, vocab_size)
        toxic_logits: 有毒模型的logits输出，同上
        wsd_p: Wasserstein距离的阶数，默认为1（Wasserstein-1距离）
        wsd_blur: Sinkhorn算法的模糊参数，控制收敛速度和精度的平衡，默认0.01
    
    Returns:
        torch.Tensor: Wasserstein切片距离值（标量）
    
    Raises:
        AssertionError: 当输入logits的维度不是2或3时抛出
    """
    assert len(base_logits.shape) in [2, 3] and len(toxic_logits.shape) in [2, 3]
    
    p = torch.nn.functional.softmax(base_logits, dim=-1)
    q = torch.nn.functional.softmax(toxic_logits, dim=-1)

    loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=wsd_p, blur=wsd_blur)
    emd = loss_fn(p, q)
    
    return emd


# 所有可用的距离/散度度量函数映射表
DiffsFuncs = {
    "none": None,           # 不使用任何度量
    "fkl": fkl_diffs_fn,    # 正向KL散度
    "rkl": rkl_diffs_fn,    # 反向KL散度
    "wsd": wsd_diffs_fn,    # Wasserstein切片距离
    "js": js_diffs_fn,      # Jensen-Shannon散度
    "tvd": tvd_diffs_fn     # 总变分距离
}

import torch
import numpy as np
import onnxruntime as ort

def compare_pt_and_onnx_models(pt_model_path, onnx_model_path):
    # 确定输入形状
    batch_size = 1
    num_observations = 15*47  # 根据您的模型输入维度

    # 创建随机输入
    dummy_input = torch.randn(batch_size, num_observations)

    # 加载 PyTorch 模型
    model_pt = torch.load(pt_model_path, map_location='cpu')
    model_pt.eval()  # 设置为评估模式

    # 使用示例输入获取 PyTorch 模型的输出
    with torch.no_grad():
        output_pt = model_pt(dummy_input)

    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(onnx_model_path)

    # ONNX Runtime 需要 numpy 数组作为输入
    dummy_input_onnx = dummy_input.numpy().astype(np.float32)

    # 使用示例输入获取 ONNX 模型的输出
    ort_inputs = {'obs': dummy_input_onnx}
    ort_outputs = ort_session.run(None, ort_inputs)
    output_onnx = ort_outputs[0]

    # 将 ONNX 输出转换为 PyTorch 张量，便于比较
    output_onnx_tensor = torch.from_numpy(output_onnx)

    # 比较两个输出（允许小的数值误差）
    comparison = torch.allclose(output_pt, output_onnx_tensor, rtol=1e-03, atol=1e-05)
    print(f"输出是否相同：{comparison}")

    if not comparison:
        # 如果输出不完全相同，打印差异
        difference = output_pt - output_onnx_tensor
        print("输出差异：")
        print(difference)

if __name__ == "__main__":
    import os

    pt_model_path = os.path.expanduser('~/pi_rl_baseline/logs/Pai_ppo/exported/policies/policy_1.pt')
    onnx_model_path = os.path.expanduser('~/pi_rl_baseline/logs/Pai_ppo/exported/policies/policy.onnx')
    compare_pt_and_onnx_models(pt_model_path, onnx_model_path)


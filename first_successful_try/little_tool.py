import argparse
import numpy as np
from pathlib import Path


def format_as_matlab_matrix(mat: np.ndarray, var_name: str = "binary_matrix") -> str:
    """
    把 2D numpy 数组格式化成 MATLAB 风格矩阵文本
    """
    mat = np.asarray(mat)

    if mat.ndim != 2:
        raise ValueError(f"输入必须是二维矩阵，当前 shape={mat.shape}")

    mat = (mat > 0.5).astype(int)

    lines = [f"{var_name} = ["]
    for i, row in enumerate(mat):
        row_str = " ".join(str(int(x)) for x in row)
        if i < mat.shape[0] - 1:
            lines.append(f"    {row_str};")
        else:
            lines.append(f"    {row_str}")
    lines.append("];")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="runs_peak_nsga2/best_pattern.npy",
        help="输入的 best_pattern.npy 路径"
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="binary_matrix",
        help="输出变量名"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：输出到 txt/m 文件；不填则只打印到终端"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")

    mat = np.load(input_path)

    # 兼容 [11,11] 或 [1,11,11]
    mat = np.asarray(mat)
    if mat.ndim == 3 and mat.shape[0] == 1:
        mat = mat[0]

    text = format_as_matlab_matrix(mat, var_name=args.var_name)

    print("\n" + text + "\n")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.write_text(text, encoding="utf-8")
        print(f"已保存到: {output_path}")


if __name__ == "__main__":
    main()
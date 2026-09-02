"""
清理预测数据文件 - 只保留第一列（xi）
"""

from pathlib import Path

# 预测数据目录
PREDICT_DIR = "./data/current/3d/CASE1/predict"

def clean_predict_file(file_path: Path):
    """只保留第一列数据"""
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 2:
            print(f"  [SKIP] 文件内容不足: {file_path.name}")
            return False

        # 处理每一行，只保留第一列
        new_lines = []
        for i, line in enumerate(lines):
            parts = line.split()
            if not parts:
                continue

            # 只保留第一列
            if i == 0:
                # 表头行
                new_lines.append(parts[0] + '\n')
            else:
                # 数据行
                new_lines.append(parts[0] + '\n')

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return True

    except Exception as e:
        print(f"  [ERROR] 处理文件失败 {file_path.name}: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("清理预测数据文件 - 只保留第一列（xi）")
    print("="*70)

    predict_path = Path(PREDICT_DIR)

    if not predict_path.exists():
        print(f"\n[ERROR] 预测数据目录不存在: {PREDICT_DIR}")
        exit(1)

    # 查找所有 txt 和 dat 文件
    txt_files = list(predict_path.rglob("*.txt")) + list(predict_path.rglob("*.dat"))

    if not txt_files:
        print(f"\n[WARN] 未找到任何 .txt 或 .dat 文件")
        exit(0)

    print(f"\n找到 {len(txt_files)} 个数据文件")
    print(f"开始处理...\n")

    success_count = 0
    for file_path in txt_files:
        print(f"处理: {file_path.relative_to(predict_path)}")
        if clean_predict_file(file_path):
            success_count += 1

    print("\n" + "="*70)
    print(f"[OK] 处理完成！")
    print("="*70)
    print(f"\n成功处理: {success_count}/{len(txt_files)} 个文件")
    print(f"\n现在所有文件只包含第一列（xi）数据")

import random
import shutil
from pathlib import Path

# =========================================================
# 1. 路径设置
# =========================================================
ROOT = Path("/root/autodl-tmp/fracatlas yolo2")

IMAGE_ROOT = ROOT / "images"
FRACTURED_DIR = IMAGE_ROOT / "Fractured"
NON_FRACTURED_DIR = IMAGE_ROOT / "Non_fractured"

LABEL_ROOT = ROOT / "YOLO"

OUT_DIR = ROOT / "datasets" / "FracAtlas_8_1_1"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42
FORCE_OVERWRITE = True

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 如果 Fractured 图片找不到标签，是否直接报错
# 正式实验建议保持 True
STRICT_FRACTURED_LABEL = True


# =========================================================
# 2. 工具函数
# =========================================================
def get_images(folder: Path):
    images = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
            images.append(p)
    return sorted(images)


def find_label_file(img_path: Path):
    """
    支持多种 YOLO 标签存放方式：

    方式 1：
    YOLO/xxx.txt

    方式 2：
    YOLO/Fractured/xxx.txt

    方式 3：
    YOLO/Non_fractured/xxx.txt

    方式 4：
    YOLO/相对于 images 的路径
    比如 images/Fractured/xxx.jpg -> YOLO/Fractured/xxx.txt
    """
    candidates = []

    # 方式 1：YOLO/xxx.txt
    candidates.append(LABEL_ROOT / f"{img_path.stem}.txt")

    # 方式 2：YOLO/Fractured/xxx.txt 或 YOLO/Non_fractured/xxx.txt
    candidates.append(LABEL_ROOT / img_path.parent.name / f"{img_path.stem}.txt")

    # 方式 3：按照相对于 images 的路径找
    try:
        rel = img_path.relative_to(IMAGE_ROOT)
        candidates.append(LABEL_ROOT / rel.with_suffix(".txt"))
    except Exception:
        pass

    for c in candidates:
        if c.exists():
            return c

    return None


def read_label(label_path: Path):
    """
    读取 YOLO 标签，并强制把类别改成 0。
    FracAtlas 这里我们只做一个类别：fracture。
    """
    if label_path is None or not label_path.exists():
        return ""

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()

    if text == "":
        return ""

    fixed_lines = []

    for line in text.splitlines():
        parts = line.strip().split()

        # YOLO 标签至少应该是：class x_center y_center width height
        if len(parts) < 5:
            continue

        # 强制类别为 0，也就是 fracture
        parts[0] = "0"

        # 只保留前 5 项，避免多余字段影响训练
        fixed_lines.append(" ".join(parts[:5]))

    return "\n".join(fixed_lines)


def split_list(items):
    """
    按 8:1:1 划分列表。
    """
    items = list(items)
    random.shuffle(items)

    n = len(items)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    return train_items, val_items, test_items


def make_unique_name(img_path: Path):
    """
    防止 Fractured 和 Non_fractured 里面存在同名图片。
    输出文件名示例：
    Fractured__xxx.jpg
    Non_fractured__xxx.jpg
    """
    rel = img_path.relative_to(IMAGE_ROOT)
    return "__".join(rel.parts)


def copy_image_and_label(img_path: Path, split: str, is_fractured: bool):
    """
    复制图片和标签到 YOLO 标准目录。
    """
    out_img_dir = OUT_DIR / "images" / split
    out_label_dir = OUT_DIR / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    out_img_name = make_unique_name(img_path)

    out_img_path = out_img_dir / out_img_name
    out_label_path = out_label_dir / Path(out_img_name).with_suffix(".txt").name

    # 复制图片
    shutil.copy2(img_path, out_img_path)

    if is_fractured:
        label_file = find_label_file(img_path)
        label_text = read_label(label_file)

        if label_text == "":
            # 理论上 Fractured 图片应该有非空标签
            out_label_path.write_text("", encoding="utf-8")
        else:
            out_label_path.write_text(label_text + "\n", encoding="utf-8")
    else:
        # Non_fractured 是负样本，必须生成空 txt
        out_label_path.write_text("", encoding="utf-8")


def count_boxes(label_path: Path):
    """
    统计一个 txt 标签文件中有多少个 bbox。
    """
    if not label_path.exists():
        return 0

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()

    if text == "":
        return 0

    return len(text.splitlines())


# =========================================================
# 3. 主程序
# =========================================================
def main():
    random.seed(SEED)

    print("=" * 80)
    print("FracAtlas YOLO 8:1:1 stratified split")
    print("=" * 80)

    print(f"ROOT:              {ROOT}")
    print(f"Fractured dir:     {FRACTURED_DIR}")
    print(f"Non_fractured dir: {NON_FRACTURED_DIR}")
    print(f"Label dir:         {LABEL_ROOT}")
    print(f"Output dir:        {OUT_DIR}")

    if not FRACTURED_DIR.exists():
        raise FileNotFoundError(f"找不到 Fractured 文件夹: {FRACTURED_DIR}")

    if not NON_FRACTURED_DIR.exists():
        raise FileNotFoundError(f"找不到 Non_fractured 文件夹: {NON_FRACTURED_DIR}")

    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"找不到 YOLO 标签文件夹: {LABEL_ROOT}")

    if OUT_DIR.exists():
        if FORCE_OVERWRITE:
            print(f"\n发现旧输出目录，正在删除: {OUT_DIR}")
            shutil.rmtree(OUT_DIR)
        else:
            raise FileExistsError(f"输出目录已存在: {OUT_DIR}")

    fractured_images = get_images(FRACTURED_DIR)
    non_fractured_images = get_images(NON_FRACTURED_DIR)

    print("\n原始数据统计：")
    print(f"Fractured 图片数:     {len(fractured_images)}")
    print(f"Non_fractured 图片数: {len(non_fractured_images)}")
    print(f"总图片数:             {len(fractured_images) + len(non_fractured_images)}")

    if len(fractured_images) == 0:
        raise RuntimeError(f"Fractured 文件夹中没有找到图片: {FRACTURED_DIR}")

    if len(non_fractured_images) == 0:
        raise RuntimeError(f"Non_fractured 文件夹中没有找到图片: {NON_FRACTURED_DIR}")

    # =====================================================
    # 4. 检查 Fractured 标签
    # =====================================================
    missing_labels = []
    empty_labels = []

    for img in fractured_images:
        label_file = find_label_file(img)

        if label_file is None:
            missing_labels.append(img)
        else:
            label_text = read_label(label_file)
            if label_text == "":
                empty_labels.append(img)

    print("\nFractured 标签检查：")
    print(f"缺失标签图片数: {len(missing_labels)}")
    print(f"空标签图片数:   {len(empty_labels)}")

    if STRICT_FRACTURED_LABEL and (len(missing_labels) > 0 or len(empty_labels) > 0):
        print("\n发现 Fractured 图片没有有效 YOLO 标签。前 20 个问题文件如下：")

        for p in missing_labels[:10]:
            print(f"[missing label] {p}")

        for p in empty_labels[:10]:
            print(f"[empty label]   {p}")

        raise RuntimeError(
            "Fractured 文件夹中存在缺失或空标签图片。"
            "请先检查 YOLO 标签路径或文件名是否对应。"
        )

    # =====================================================
    # 5. 分层 8:1:1 划分
    # =====================================================
    f_train, f_val, f_test = split_list(fractured_images)
    n_train, n_val, n_test = split_list(non_fractured_images)

    split_data = {
        "train": {
            "fractured": f_train,
            "non_fractured": n_train,
        },
        "val": {
            "fractured": f_val,
            "non_fractured": n_val,
        },
        "test": {
            "fractured": f_test,
            "non_fractured": n_test,
        },
    }

    print("\n划分结果：")
    for split in ["train", "val", "test"]:
        f_num = len(split_data[split]["fractured"])
        n_num = len(split_data[split]["non_fractured"])
        total = f_num + n_num

        print(
            f"{split}: total={total}, "
            f"fractured={f_num}, "
            f"non_fractured={n_num}"
        )

    # =====================================================
    # 6. 复制图片和标签
    # =====================================================
    print("\n正在复制图片和标签...")

    for split in ["train", "val", "test"]:
        for img in split_data[split]["fractured"]:
            copy_image_and_label(img, split, is_fractured=True)

        for img in split_data[split]["non_fractured"]:
            copy_image_and_label(img, split, is_fractured=False)

    # =====================================================
    # 7. 生成 YOLO data yaml
    # =====================================================
    yaml_path = OUT_DIR / "fracatlas.yaml"

    yaml_text = f"""# FracAtlas YOLO dataset
path: {OUT_DIR}
train: images/train
val: images/val
test: images/test

names:
  0: fracture
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("\n已生成 YOLO 配置文件：")
    print(yaml_path)

    # =====================================================
    # 8. 最终检查
    # =====================================================
    print("\n最终文件检查：")

    for split in ["train", "val", "test"]:
        img_dir = OUT_DIR / "images" / split
        label_dir = OUT_DIR / "labels" / split

        image_files = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
        ]

        label_files = list(label_dir.glob("*.txt"))

        positive_label_files = 0
        total_boxes = 0

        for label in label_files:
            box_num = count_boxes(label)

            if box_num > 0:
                positive_label_files += 1
                total_boxes += box_num

        negative_label_files = len(label_files) - positive_label_files

        print(
            f"{split}: "
            f"images={len(image_files)}, "
            f"labels={len(label_files)}, "
            f"positive_label_files={positive_label_files}, "
            f"negative_label_files={negative_label_files}, "
            f"boxes={total_boxes}"
        )

    print("\n完成！")
    print("=" * 80)
    print("训练时 data 参数使用：")
    print(f'data="{yaml_path}"')
    print("=" * 80)


if __name__ == "__main__":
    main()
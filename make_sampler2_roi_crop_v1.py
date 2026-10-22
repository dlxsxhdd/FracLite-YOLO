from pathlib import Path
import shutil
import yaml


# ============================================================
# 1. 基础路径
# ============================================================

ROOT = Path("/root/autodl-tmp/fracatlas yolo2")

# 原始固定 8:1:1 数据集
BASE_ROOT = ROOT / "datasets" / "FracAtlas_8_1_1"

# 已经生成好的 ROI crop v1 数据集
ROI_ROOT = ROOT / "datasets" / "FracAtlas_roi_crop_v1"

# 新建：
# sampler2 + ROI crop
OUT_ROOT = ROOT / "datasets" / "FracAtlas_sampler2_roi_crop_v1"

IMG_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. 工具函数
# ============================================================

def get_images(folder: Path):
    """返回目录中的所有图像文件。"""
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def find_image_by_stem(folder: Path, stem: str):
    """根据文件 stem 查找对应图片。"""
    for p in folder.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in IMG_EXTS
            and p.stem == stem
        ):
            return p

    return None


def read_label(label_path: Path):
    """读取YOLO标签。"""
    if not label_path.exists():
        return ""

    return label_path.read_text(
        encoding="utf-8"
    ).strip()


def count_boxes(label_path: Path):
    """统计一个标签文件中的bbox数量。"""
    text = read_label(label_path)

    if not text:
        return 0

    return len(
        [
            line
            for line in text.splitlines()
            if line.strip()
        ]
    )


def make_dirs():
    """创建新数据集目录结构。"""

    if OUT_ROOT.exists():
        raise FileExistsError(
            f"\n输出目录已经存在：\n{OUT_ROOT}\n\n"
            "为避免覆盖已有实验，本脚本已停止。\n"
            "如果确认旧目录可以删除，请手动删除后重新运行：\n"
            f'rm -rf "{OUT_ROOT}"'
        )

    for split in ["train", "val", "test"]:
        (OUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def copy_split(src_root: Path, split: str):
    """
    将一个完整split复制到OUT。
    """

    src_img_dir = src_root / "images" / split
    src_lab_dir = src_root / "labels" / split

    dst_img_dir = OUT_ROOT / "images" / split
    dst_lab_dir = OUT_ROOT / "labels" / split

    if not src_img_dir.exists():
        raise FileNotFoundError(src_img_dir)

    if not src_lab_dir.exists():
        raise FileNotFoundError(src_lab_dir)

    images = get_images(src_img_dir)

    for img_path in images:

        dst_img = dst_img_dir / img_path.name

        shutil.copy2(
            img_path,
            dst_img,
        )

        label_path = (
            src_lab_dir /
            f"{img_path.stem}.txt"
        )

        dst_label = (
            dst_lab_dir /
            f"{img_path.stem}.txt"
        )

        if label_path.exists():
            shutil.copy2(
                label_path,
                dst_label,
            )
        else:
            # 没有标签时创建空标签
            dst_label.write_text(
                "",
                encoding="utf-8",
            )


def dataset_stats(root: Path, split: str):

    img_dir = root / "images" / split
    lab_dir = root / "labels" / split

    images = get_images(img_dir)

    positive = 0
    negative = 0
    boxes = 0
    missing_labels = 0

    for img_path in images:

        label_path = (
            lab_dir /
            f"{img_path.stem}.txt"
        )

        if not label_path.exists():
            missing_labels += 1
            negative += 1
            continue

        n_boxes = count_boxes(label_path)

        if n_boxes > 0:
            positive += 1
            boxes += n_boxes
        else:
            negative += 1

    return {
        "images": len(images),
        "positive": positive,
        "negative": negative,
        "boxes": boxes,
        "missing_labels": missing_labels,
    }


# ============================================================
# 3. 主程序
# ============================================================

def main():

    print("=" * 75)
    print("Building FracAtlas_sampler2_roi_crop_v1")
    print("=" * 75)

    print("\n[Input datasets]")
    print(f"Original 8:1:1 : {BASE_ROOT}")
    print(f"ROI crop v1    : {ROI_ROOT}")
    print(f"Output         : {OUT_ROOT}")

    # --------------------------------------------------------
    # Step 1：检查原始数据
    # --------------------------------------------------------

    if not BASE_ROOT.exists():
        raise FileNotFoundError(
            f"Base dataset not found:\n{BASE_ROOT}"
        )

    if not ROI_ROOT.exists():
        raise FileNotFoundError(
            f"ROI dataset not found:\n{ROI_ROOT}"
        )

    print("\n[Original dataset statistics]")

    for split in ["train", "val", "test"]:

        stats = dataset_stats(
            BASE_ROOT,
            split,
        )

        print(
            f"{split:5s}: "
            f"images={stats['images']}, "
            f"positive={stats['positive']}, "
            f"negative={stats['negative']}, "
            f"boxes={stats['boxes']}"
        )

    print("\n[Existing ROI dataset statistics]")

    for split in ["train", "val", "test"]:

        stats = dataset_stats(
            ROI_ROOT,
            split,
        )

        print(
            f"{split:5s}: "
            f"images={stats['images']}, "
            f"positive={stats['positive']}, "
            f"negative={stats['negative']}, "
            f"boxes={stats['boxes']}"
        )

    # --------------------------------------------------------
    # Step 2：创建新目录
    # --------------------------------------------------------

    print("\nCreating output directories...")

    make_dirs()

    # --------------------------------------------------------
    # Step 3：
    # 复制ROI crop v1全部train
    #
    # 此时train应为：
    # 原始完整图 + 1498张ROI
    # --------------------------------------------------------

    print("\nCopying ROI crop v1 training set...")

    copy_split(
        ROI_ROOT,
        "train",
    )

    # --------------------------------------------------------
    # Step 4：
    # 从原始8:1:1 train中，
    # 对每一张阳性完整X线图再复制一次。
    #
    # 注意：
    # 这里只重复完整图，
    # 不重复ROI crop。
    # --------------------------------------------------------

    print("\nApplying positive resampling (2x full-image positives)...")

    base_train_img_dir = (
        BASE_ROOT /
        "images" /
        "train"
    )

    base_train_lab_dir = (
        BASE_ROOT /
        "labels" /
        "train"
    )

    out_train_img_dir = (
        OUT_ROOT /
        "images" /
        "train"
    )

    out_train_lab_dir = (
        OUT_ROOT /
        "labels" /
        "train"
    )

    duplicated_images = 0
    duplicated_boxes = 0

    label_files = sorted(
        base_train_lab_dir.glob("*.txt")
    )

    for label_path in label_files:

        text = read_label(label_path)

        # 空标签代表非骨折图片
        if not text:
            continue

        image_path = find_image_by_stem(
            base_train_img_dir,
            label_path.stem,
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Positive image not found for label:\n"
                f"{label_path}"
            )

        # 新文件名避免与原图冲突
        new_stem = (
            f"{image_path.stem}"
            f"__sampler2dup"
        )

        new_img_path = (
            out_train_img_dir /
            f"{new_stem}{image_path.suffix}"
        )

        new_label_path = (
            out_train_lab_dir /
            f"{new_stem}.txt"
        )

        if new_img_path.exists():
            raise FileExistsError(
                f"Duplicate output image exists:\n"
                f"{new_img_path}"
            )

        shutil.copy2(
            image_path,
            new_img_path,
        )

        shutil.copy2(
            label_path,
            new_label_path,
        )

        duplicated_images += 1

        duplicated_boxes += count_boxes(
            label_path
        )

    print(
        f"Duplicated positive images: "
        f"{duplicated_images}"
    )

    print(
        f"Duplicated fracture boxes: "
        f"{duplicated_boxes}"
    )

    # --------------------------------------------------------
    # Step 5：
    # val/test严格保持不变
    #
    # 从ROI crop数据集复制即可，
    # 因为它的val/test原本就是完整影像。
    # --------------------------------------------------------

    print("\nCopying unchanged validation set...")

    copy_split(
        ROI_ROOT,
        "val",
    )

    print("Copying unchanged test set...")

    copy_split(
        ROI_ROOT,
        "test",
    )

    # --------------------------------------------------------
    # Step 6：生成YAML
    # --------------------------------------------------------

    yaml_data = {
        "path": str(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "fracture"
        }
    }

    yaml_path = (
        OUT_ROOT /
        "fracatlas.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            yaml_data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    # --------------------------------------------------------
    # Step 7：最终统计
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL DATASET STATISTICS")
    print("=" * 75)

    all_stats = {}

    for split in [
        "train",
        "val",
        "test",
    ]:

        stats = dataset_stats(
            OUT_ROOT,
            split,
        )

        all_stats[split] = stats

        print(
            f"{split:5s}: "
            f"images={stats['images']}, "
            f"positive={stats['positive']}, "
            f"negative={stats['negative']}, "
            f"boxes={stats['boxes']}, "
            f"missing_labels="
            f"{stats['missing_labels']}"
        )

    # --------------------------------------------------------
    # Step 8：严格检查预期数字
    # --------------------------------------------------------

    expected = {
        "train": {
            "images": 5289,
            "positive": 2644,
            "negative": 2645,
            "boxes": 3431,
        },
        "val": {
            "images": 401,
            "positive": 71,
            "negative": 330,
            "boxes": 86,
        },
        "test": {
            "images": 405,
            "positive": 73,
            "negative": 332,
            "boxes": 87,
        },
    }

    print("\nChecking expected statistics...")

    problems = []

    for split in expected:

        for key in expected[split]:

            actual = all_stats[split][key]

            target = expected[split][key]

            if actual != target:

                problems.append(
                    f"{split}.{key}: "
                    f"actual={actual}, "
                    f"expected={target}"
                )

    if problems:

        print("\nERROR: dataset statistics mismatch!")

        for p in problems:
            print("  -", p)

        raise RuntimeError(
            "\nDataset construction completed "
            "but statistics are incorrect. "
            "DO NOT START TRAINING."
        )

    print("\nAll statistics are correct.")

    print("\nDataset YAML:")
    print(yaml_path)

    print("\nDataset root:")
    print(OUT_ROOT)

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()
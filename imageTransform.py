import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = "test_image.png"   # غيّر الاسم هنا لو الصورة اسمها مختلف


# ----------------------------------------------------------------------
# Part 1: الصورة = مصفوفة (Matrix)
# ----------------------------------------------------------------------
def part1_image_as_matrix(path):
    img = Image.open(path).convert("RGB")   # نضمن إنها 3 قنوات RGB
    A = np.array(img)                       # نحول الصورة لمصفوفة numpy

    print("Shape (height, width, channels):", A.shape)
    print("Pixel [20, 20]   =", A[20, 20])    # صف 20، عمود 20 -> [R, G, B]
    if A.shape[0] > 100 and A.shape[1] > 160:
        print("Pixel [100, 160] =", A[100, 160])
    return A


# ----------------------------------------------------------------------
# Part 2: تحويل شكل F بمصفوفات
# ----------------------------------------------------------------------
# كل عمود في F هو نقطة (x, y). الصف الأول x والتاني y.
F = np.array([[0, 1, 1, 2.5, 2.5, 1, 1, 3, 3, 0],
              [0, 0, 2, 2,   3,   3, 4, 4, 5, 5]])

# مصفوفات التحويل (2x2)
transformations = {
    "Rotate 90° clockwise":   np.array([[0, 1], [-1, 0]]),
    "Rotate 90° counter-cw":  np.array([[0, -1], [1, 0]]),
    "Reflect over x-axis":    np.array([[1, 0], [0, -1]]),
    "Reflect over y-axis":    np.array([[-1, 0], [0, 1]]),
    "Scale (x2 in x, x0.5 y)": np.array([[2, 0], [0, 0.5]]),
    "Shear (horizontal)":     np.array([[1, 0.5], [0, 1]]),
}


def rotation_matrix(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)],
                     [np.sin(t),  np.cos(t)]])


def part2_transform_F():
    n = len(transformations)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, M) in zip(axes.ravel(), transformations.items()):
        G = M @ F                                  # قلب الموضوع: ضرب مصفوفات
        ax.fill(F[0], F[1], alpha=0.4, label="original")
        ax.fill(G[0], G[1], alpha=0.4, label="after")
        ax.axhline(0, color="black")
        ax.axvline(0, color="black")
        ax.axis("equal")
        ax.grid(True)
        ax.set_title(name)
        ax.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Part 3: تحويل صورة حقيقية
# ----------------------------------------------------------------------
def transform_image(A, M):
    """
    بنطبّق المصفوفة M على كل بكسل في الصورة.
    بنستخدم طريقة "inverse mapping": لكل بكسل في الصورة الناتجة
    بنسأل: ده جه من أنهي بكسل في الصورة الأصلية؟ (عن طريق M^-1)
    ده بيمنع ظهور ثقوب في الصورة الناتجة.
    """
    h, w = A.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2         # مركز الصورة (نقطة الأصل)

    # 1) نحسب حجم الصورة الناتجة من تحويل الأركان الأربعة
    corners = np.array([[-cx, cx, cx, -cx],
                        [ cy, cy, -cy, -cy]])
    new = M @ corners
    xmin, xmax = new[0].min(), new[0].max()
    ymin, ymax = new[1].min(), new[1].max()
    W = int(np.ceil(xmax - xmin)) + 1
    H = int(np.ceil(ymax - ymin)) + 1

    # 2) إحداثيات كل بكسل في الصورة الناتجة (y لفوق زي الرسم الرياضي)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    P = np.vstack([(cols + xmin).ravel(), (ymax - rows).ravel()])

    # 3) نرجع للصورة الأصلية بالـ inverse
    S = np.linalg.inv(M) @ P
    src_col = np.rint(S[0] + cx).astype(int)
    src_row = np.rint(cy - S[1]).astype(int)

    valid = (src_col >= 0) & (src_col < w) & (src_row >= 0) & (src_row < h)

    # 4) ننسخ قيم البكسلات
    out = np.zeros((H, W) + A.shape[2:], dtype=A.dtype)
    flat = out.reshape(-1, *A.shape[2:])
    flat[valid] = A[src_row[valid], src_col[valid]]
    return out


def part3_transform_image(A):
    examples = {
        "Original": np.eye(2),
        "Rotate 45°": rotation_matrix(45),
        "Reflect (y-axis)": np.array([[-1, 0], [0, 1]]),
        "Scale (x1.5 , x0.7)": np.array([[1.5, 0], [0, 0.7]]),
        "Shear": np.array([[1, 0.5], [0, 1]]),
        "Rotate 90° clockwise": np.array([[0, 1], [-1, 0]]),
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, M) in zip(axes.ravel(), examples.items()):
        ax.imshow(transform_image(A, M))
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    A = part1_image_as_matrix(IMAGE_PATH)
    part2_transform_F()
    part3_transform_image(A)

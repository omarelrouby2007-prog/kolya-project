import numpy as np  # matrixالمكتبة الخاصه بال 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection        #ده علشان يرسم الشكل
from PIL import Image  #هنا علشان احط  الصورة
IMAGE_PATH = "test_image.png" #ده اسم الصورة قلي انا منزلها
ANGLE_X = 0   # لفة حوالين محور x بالدرجات
ANGLE_Y = 0  # لفة حوالين محور y بالدرجات
ANGLE_Z = 0  # لفة حوالين محور z، وهي نفس لفة الـ 2D
SX = 1
SY = 1
SZ = 1       # الجديد: scaling في اتجاه z

FLIP = np.array([[1,  0, 0],
                 [0, 1, 0],
                 [0,  0, 1]])  # بقت 3*3 علشان ده 3D، بتقلب y بس وx وz زي ما هم
              # مينفعش يكون العمود في 3 زيرو علشان كده   Determinat=0 no inverse

F = np.array([[0, 1, 1, 2.5, 2.5, 1, 1, 3, 3, 0], # هنا بقت 10*3، الصف التالت هو z وكله صفر يعني الحرف مسطح
              [0, 0, 2, 2,   3,   3, 4, 4, 5, 5],
              [0, 0, 0, 0,   0,   0, 0, 0, 0, 0]])
def load_image(path):#الدالة دي بتاخد الpath الصورة
    A = np.array(Image.open(path).convert("RGB"))#بيفتح ملف الصورةوبيحولها لمصفوفةوصورةبالالوان من غيره الصورة هتكون ابيض واسود
    print("Image shape (h, w, channels):", A.shape)#بيطبع ارتفاع وعرض الصورة
    print("Pixel [20, 20] =", A[20, 20])#دي المفروض انت حاتها علشان RGB[90,90,90]
    return A


def rotation_x(theta_deg):  # بتلف حوالين x، فالصف والعمود بتوع x ما بيتغيروش
    t = np.radians(theta_deg)   # numpy بتشتغل بالراديان مش بالدرجات
    return np.array([[1, 0,          0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t),  np.cos(t)]])


def rotation_y(theta_deg):  # بتلف حوالين y، فالصف والعمود بتوع y ما بيتغيروش
    t = np.radians(theta_deg)
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [ 0,         1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def rotation_z(theta_deg):  # نفس مصفوفة الـ 2D، بس زودنا صف وعمود لـ z
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0,          0,         1]])


def scaling_matrix(sx, sy, sz):
    return np.array([[sx, 0,  0],
                     [0,  sy, 0],
                     [0,  0,  sz]])



def change_colors(A, color=(0, 255, 0)):
    B = A.copy()
    inner = B[15:-15, 15:-15]
    mask = (inner < 30).all(axis=2)
    inner[mask] = color
    return B




def plot_F(M):#بتاخد المصفوفة Mدي بتاعت ال3ديه وبتضربها في الحرف
    G = M @ F #هنا صربنا الاتنين مع بعد
    fig = plt.figure() #احنا عرفنا ماتلاب فوق علشان و خلينا اسموابplt وبيعمل حاجة زي رسم بيانيه
    ax = fig.add_subplot(projection="3d") # الجديد: محاور 3D بدل المحاور العادية
    ax.add_collection3d(Poly3DCollection([list(zip(*F))], alpha=0.5, facecolor="blue"))   # original
    ax.add_collection3d(Poly3DCollection([list(zip(*G))], alpha=0.5, facecolor="orange")) # after#الصف الاول x والتاني y والتالت z قبل الضرب في 3ديه
    #بعد ضرب
    m = max(np.abs(F).max(), np.abs(G).max()) # أكبر قيمة علشان نساوي المقياس في المحاور التلاتة
    ax.set_xlim(-m, m)#بيخلّي مقياس x وy وz متساويين، يعني 1 على اي محور بنفس الطول. من غيره
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_box_aspect((1, 1, 1)) # بيخلي الصندوق مكعب
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("F shape")# بيكتب فوق الرسمها عنوانها

def transform_image(A, M):#لفانكشن بتاخد صورتك A والمصفوفة M
    M = M[:2, :2] # الجديد: الصورة 2D، فبناخد الجزء 2*2 بس من M
    h, w = A.shape[:2] #ده بيخلني اخد قمتين من الصورة الطول و العرض
    cx, cy = (w - 1) / 2, (h - 1) / 2     #وهنا انا بحسب النقطة    

    corners = np.array([[-cx, cx, cx, -cx], #فيها الأركان الأربعة للصورة، كل عمود ركن (x, y)، زي F بالظبط. وبما إن المركز بقى (0,0)، الأركا
                        [ cy, cy, -cy, -cy]])
    new = M @ corners #بنطبّق M على الأركان الأربعة بس، مش على ملايين البكسلات.
    xmin, xmax = new[0].min(), new[0].max() #بيجيبوا أصغر وأكبر قيمة، فنعرف الصورة الجديدة بتمتد من فين لفين
    ymin, ymax = new[1].min(), new[1].max()
    W = int(np.ceil(xmax - xmin)) + 1#هما عرض وارتفاع الصورة الجديدة
    H = int(np.ceil(ymax - ymin)) + 1
    cols, rows = np.meshgrid(np.arange(W), np.arange(H)) 

 #np.arange(W) بيعمل الأرقام 0، 1، 2، ... لحد W-1.
 #np.meshgrid بيعمل شبكة: cols فيها رقم العمود لكل بكسل، وrows فيها رقم الصف.
 #cols + xmin بيحوّل رقم العمود لإحداثي x في المحاور اللي المركز فيها هو الأصل. فالعمود 0 هو أقصى
#ymax - rows نفس الفكرة لـ y، بس مقلوبة، لأن الصفوف بتزيد لتحت لكن y الرياضي بيزيد لفوق
#.ravel() بيفرد المصفوفة في صف واحد طويل.
#np.vstack بيرصّ الصفين فوق بعض، فتطلع P بشكل 2×(عدد البكسلات). وده نفس شكل F، كل عمود
    P = np.vstack([(cols + xmin).ravel(), (ymax - rows).ravel()])
    S = np.linalg.inv(M) @ P
    src_col = np.rint(S[0] + cx).astype(int)
    src_row = np.rint(cy - S[1]).astype(int)
    valid = (src_col >= 0) & (src_col < w) & (src_row >= 0) & (src_row < h)
    #np.linalg.inv(M) هو M⁻¹ اللي خدته في الكلاس
    #S = M⁻¹ @ P: لكل بكسل جديد (P) بنضربه في الـ inverse، فيطلع مكانه في الصورة الأصلية. ولو M
    #ملهاش inverse (det = 0) السطر ده هو اللي بيرمي Singular matrix
    #+ cx وcy - بيرجّعوا الإحداثيات من "المركز هو الأصل" لأرقام صف وعمود الصورة العادية
    out = np.zeros((H, W) + A.shape[2:], dtype=A.dtype)
    flat = out.reshape(-1, *A.shape[2:])
    flat[valid] = A[src_row[valid], src_col[valid]]
    return out


def plot_image(A, M):#الصورة القديمة
    B = transform_image(A, M)#الصورة يعد التعديل
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) #بيعمل نافذة فيها صف واحد وعمودين، يعني مكانين جنب بعض.
    axes[0].imshow(A)
    axes[0].set_title("Original")
    axes[1].imshow(B)
    axes[1].set_title("After")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

if __name__ == "__main__":
    R = rotation_z(ANGLE_Z) @ rotation_y(ANGLE_Y) @ rotation_x(ANGLE_X) # بنلف حوالين x وبعدين y وبعدين z
    S = scaling_matrix(SX, SY, SZ)
    M = R @ S @ FLIP # بيتقري من اليمين: flip وبعدين scale وبعدين لفة
    print("M =\n", M)
    A = load_image(IMAGE_PATH)
    A = change_colors(A, (0, 0, 0))
    plot_F(M)
    plot_image(A, M)
    plt.show()

import numpy as np  # matrixالمكتبة الخاصه بال 
import matplotlib.pyplot as plt #ده علشان يرسم الشكل
from PIL import Image  #هنا علشان احط  الصورة
IMAGE_PATH = "test_image.png" #ده اسم الصورة قلي انا منزلها

M = np.array([[1,  0], #   x,yركزوا معايا هنا اللعب كله لية علشان لو خلتها زي ما هي كده هي نحيه الموجابة من 
              [0, -1]] )#يعني حرف ال فيfirst quad 
              # طب ليه 2*2 علشان ده 2ديه 
              # مينفعش يكون العمود في 2 زيرو علشان كده   Determinat=0 no inverse       

F = np.array([[0, 1, 1, 2.5, 2.5, 1, 1, 3, 3, 0], # هنا 10*2 هتكون موجودة في الصور قلي بعتها
              [0, 0, 2, 2,   3,   3, 4, 4, 5, 5]])
def load_image(path):#الدالة دي بتاخد الpath الصورة
    A = np.array(Image.open(path).convert("RGB"))#بيفتح ملف الصورةوبيحولها لمصفوفةوصورةبالالوان من غيره الصورة هتكون ابيض واسود
    print("Image shape (h, w, channels):", A.shape)#بيطبع ارتفاع وعرض الصورة
    print("Pixel [20, 20] =", A[20, 20])#دي المفروض انت حاتها علشان RGB[90,90,90]
    return A



def change_colors(A, color=(0, 255, 0)):
    B = A.copy()
    inner = B[15:-15, 15:-15]
    mask = (inner < 30).all(axis=2)
    inner[mask] = color
    return B




def plot_F(M):#بتاخد المصفوفة Mدي بتاعت ال2ديه وبتضربها في الحرق
    G = M @ F #هنا صربنا الاتنين مع بعد
    plt.figure() #احنا عرفنا ماتلاب فوق علشان و خلينا اسموابplt وبيعمل حاجة زي رسم بيانيه
    plt.fill(F[0], F[1], alpha=0.5, label="original")#الصفر ده الصف بتاع الحرف والواحد ده العمود قبل الضرب في2ديه
    plt.fill(G[0], G[1], alpha=0.5, label="after")#بعد ضرب
    plt.axhline(0, color="black") #horizontal line at y=0 and it`color is balck
    plt.axvline(0, color="black")# vertical line ay x=0 and it`s color is black
    plt.axis("equal")#بيخلّي مقياس x ومقياس y متساويين، يعني 1 على المحور الأفقي بنفس طول 1 على الرأسي. من غيره
    plt.grid(True)#بيرسم شبكة مربعات في الخلفية
    plt.legend()
    plt.title("F shape")# بيكتب فوق الرسمها عنوانها

def transform_image(A, M):#لفانكشن بتاخد صورتك A والمصفوفة M
    h, w = A.shape[:2] #ده بيخلني اخد قمتين من الصورة الطول و العرض
    cx, cy = (w - 1) / 2, (h - 1) / 2       #وهنا انا بحسب النقطة    

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
    print("M =\n", M)
    A = load_image(IMAGE_PATH)
    A = change_colors(A, (255, 0, 0))
    plot_F(M)
    plot_image(A, M)
    plt.show()


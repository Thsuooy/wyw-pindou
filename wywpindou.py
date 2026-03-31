import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import cv2
from rembg import remove, new_session

# ================= 1. 官方真实色板 (Perler Standard) =================
# 替换为真实的 Perler Beads (帕尔) 标准物理色号，可直接网购对应型号
PERLER_COLORS = {
    "P01": {"name": "White (白色)", "rgb": (255, 255, 255)},
    "P02": {"name": "Cream (奶油)", "rgb": (235, 235, 204)},
    "P03": {"name": "Yellow (黄色)", "rgb": (240, 194, 21)},
    "P04": {"name": "Orange (橙色)", "rgb": (240, 116, 45)},
    "P05": {"name": "Red (大红)", "rgb": (201, 36, 49)},
    "P06": {"name": "Bubblegum (泡泡糖粉)", "rgb": (225, 107, 137)},
    "P07": {"name": "Purple (紫色)", "rgb": (111, 56, 137)},
    "P08": {"name": "Dark Blue (深蓝)", "rgb": (41, 62, 137)},
    "P09": {"name": "Light Blue (浅蓝)", "rgb": (72, 165, 219)},
    "P10": {"name": "Dark Green (深绿)", "rgb": (43, 94, 62)},
    "P11": {"name": "Light Green (浅绿)", "rgb": (86, 183, 121)},
    "P12": {"name": "Brown (深棕)", "rgb": (94, 62, 46)},
    "P17": {"name": "Light Grey (浅灰)", "rgb": (140, 144, 145)},
    "P18": {"name": "Black (黑色)", "rgb": (0, 0, 0)},
    "P20": {"name": "Rust (锈红)", "rgb": (158, 66, 42)},
    "P21": {"name": "Light Brown (浅棕)", "rgb": (153, 102, 51)},
    "P22": {"name": "Peach (肤色/桃粉)", "rgb": (244, 180, 154)},
    "P35": {"name": "Tan (黄褐色/沙色)", "rgb": (204, 163, 114)},
    "P38": {"name": "Magenta (品红)", "rgb": (202, 53, 116)},
    "P43": {"name": "Pastel Yellow (淡黄)", "rgb": (245, 230, 120)},
    "P47": {"name": "Pastel Green (柔绿)", "rgb": (132, 203, 145)},
    "P52": {"name": "Pastel Blue (柔蓝)", "rgb": (108, 163, 212)},
    "P53": {"name": "Pastel Lavender (淡紫)", "rgb": (180, 150, 200)},
    "P56": {"name": "Plum (李子紫)", "rgb": (142, 65, 143)},
    "P57": {"name": "Cheddar (奶酪橙)", "rgb": (243, 152, 44)},
    "P58": {"name": "Toothpaste (绿松石蓝)", "rgb": (58, 168, 193)},
    "P60": {"name": "Plum/Cherry (樱桃红)", "rgb": (180, 40, 60)},
    "P80": {"name": "Artic Blue (极地蓝)", "rgb": (116, 149, 191)},
    "P83": {"name": "Pink (粉红)", "rgb": (234, 113, 157)},
    "P88": {"name": "Raspberry (蔓越莓红)", "rgb": (173, 50, 87)},
    "P90": {"name": "Butterscotch (奶油糖)", "rgb": (216, 145, 58)},
    "P92": {"name": "Dark Grey (深灰)", "rgb": (84, 88, 90)},
    "P101": {"name": "Light Pink (浅粉)", "rgb": (245, 190, 211)}
}

# ================= 2. 核心图像处理函数 =================
def apply_palette_and_dither(img_small, alpha_array, palette, use_dither, alpha_threshold):
    """使用最有效的调色板量化与抖动算法"""
    keys = list(palette.keys())

    # 按照 Pillow 的要求展平色板
    flat_palette = []
    for k in keys:
        flat_palette.extend(palette[k]["rgb"])
    # 填充到 256 色（768个数值），Pillow 引擎的硬性要求
    flat_palette.extend([0] * (768 - len(flat_palette)))

    # 创建一个标准色板图像
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat_palette)

    # 强制转换原图为 RGB 并进行调色板量化匹配
    img_rgb = img_small.convert("RGB")
    dither_mode = Image.FLOYDSTEINBERG if use_dither else Image.NONE

    # 核心魔法：使用引擎极速匹配最近颜色，并自动进行视觉抖动
    img_quant = img_rgb.quantize(palette=pal_img, dither=dither_mode)
    quant_array = np.array(img_quant)

    h, w = quant_array.shape
    mapped_pixels = np.zeros((h, w, 3), dtype=np.uint8)
    code_matrix = np.empty((h, w), dtype=object)

    for y in range(h):
        for x in range(w):
            # 处理透明背景
            if alpha_array[y, x] < alpha_threshold:
                mapped_pixels[y, x] = [255, 255, 255]
                code_matrix[y, x] = None
            else:
                idx = quant_array[y, x]
                # 防止索引越界（极小概率）
                if idx < len(keys):
                    code = keys[idx]
                    mapped_pixels[y, x] = palette[code]["rgb"]
                    code_matrix[y, x] = code
                else:
                    mapped_pixels[y, x] = [255, 255, 255]
                    code_matrix[y, x] = None

    return mapped_pixels, code_matrix


def create_blueprint(image, max_size, palette, use_ai_bg_removal, ai_model_name, use_clahe, alpha_threshold, brightness,
                     use_dither):
    """图纸生成逻辑"""
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    if use_ai_bg_removal:
        session = new_session(ai_model_name)
        image = remove(image, session=session)
    else:
        image = image.convert("RGBA")

    img_cv_temp = np.array(image)
    if img_cv_temp.shape[2] == 4:
        alpha_layer = img_cv_temp[:, :, 3]
        coords = cv2.findNonZero(alpha_layer)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            x, y = max(0, x - 1), max(0, y - 1)
            w = min(img_cv_temp.shape[1] - x, w + 2)
            h = min(img_cv_temp.shape[0] - y, h + 2)
            image = image.crop((x, y, x + w, y + h))

    img_cv = np.array(image)

    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:
        alpha_channel = img_cv[:, :, 3]
        rgb_channels = img_cv[:, :, :3]
    else:
        alpha_channel = np.ones(img_cv.shape[:2], dtype=np.uint8) * 255
        rgb_channels = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB) if len(img_cv.shape) == 2 else img_cv[:, :, :3]

    if use_clahe:
        lab = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        rgb_channels = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    rgb_channels = cv2.bilateralFilter(rgb_channels, d=9, sigmaColor=70, sigmaSpace=70)

    processed_cv = np.dstack((rgb_channels, alpha_channel))
    processed_pil = Image.fromarray(processed_cv)

    processed_pil = processed_pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    orig_w, orig_h = processed_pil.size
    if orig_w > orig_h:
        new_w = max_size
        new_h = max(1, int(max_size * (orig_h / orig_w)))
    else:
        new_h = max_size
        new_w = max(1, int(max_size * (orig_w / orig_h)))

    img_small = processed_pil.resize((new_w, new_h), Image.Resampling.BOX)
    alpha_array = np.array(img_small.split()[-1]) if img_small.mode == "RGBA" else np.ones((new_h, new_w)) * 255

    mapped_array, code_matrix = apply_palette_and_dither(img_small, alpha_array, palette, use_dither, alpha_threshold)

    h, w, _ = mapped_array.shape
    fig, ax = plt.subplots(figsize=(w * 0.4, h * 0.4), dpi=150)
    ax.imshow(mapped_array)

    ax.set_xticks(np.arange(w))
    ax.set_xticklabels(np.arange(1, w + 1), fontsize=8, color="#555555")
    ax.set_yticks(np.arange(h))
    ax.set_yticklabels(np.arange(1, h + 1), fontsize=8, color="#555555")
    ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, left=True, labelleft=True, right=True,
                   labelright=True)

    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linestyle='-', linewidth=0.8)
    ax.grid(which="major", visible=False)

    for i in range(10, w, 10):
        ax.axvline(x=i - 0.5, color='#d93838', linewidth=1.5, alpha=0.9)
    for i in range(10, h, 10):
        ax.axhline(y=i - 0.5, color='#d93838', linewidth=1.5, alpha=0.9)

    bead_counts = {}
    for y in range(h):
        for x in range(w):
            code = code_matrix[y, x]
            if code is not None:
                bead_counts[code] = bead_counts.get(code, 0) + 1
                rgb = palette[code]["rgb"]
                luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                text_color = "white" if luminance < 130 else "black"
                font_s = 8 if max_size <= 40 else (6 if max_size <= 80 else 4)
                ax.text(x, y, code, ha='center', va='center', color=text_color, fontsize=font_s, fontweight='bold')

    plt.tight_layout()
    return fig, bead_counts


# ================= 3. 网页界面设计 =================
st.set_page_config(page_title="wyw专属拼豆图纸生成器", page_icon="🎨", layout="wide")
st.title("🎨 wyw专属拼豆图纸生成器")
st.write("提供全套真实官方物理色卡匹配、智能主体识别与边缘优化功能。")

st.sidebar.header("参数设置")
grid_size = st.sidebar.slider("图纸精细度 (最长边豆子数)", min_value=15, max_value=120, value=50, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("色彩校正 (解决颜色不准)")
st.sidebar.info("💡 **小贴士**：如果白色物体生成出来发黄、发灰或变成了棕色，请调高曝光亮度！")
brightness_val = st.sidebar.slider("曝光亮度补偿", min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                                   help="1.0为原图。调高可消除阴影，让物体更白亮。")
use_dither = st.sidebar.toggle("开启像素抖动 (解决色彩断层)", value=True,
                               help="当颜色缺失时，用两种相近颜色的豆子交替排列来欺骗视觉，效果更逼真！")

st.sidebar.markdown("---")
st.sidebar.subheader("AI 抠图与画质优化")
use_ai = st.sidebar.toggle("开启 AI 智能抠图", value=True)

model_choice = st.sidebar.selectbox(
    "选择 AI 抠图模型",
    [
        "静物与通用主体 (isnet-general) - 推荐",
        "通用快速模型 (u2net)",
        "动漫插画专属 (isnet-anime)",
        "真实人物全身 (u2net_human_seg)"
    ],
    index=0
)
model_dict = {
    "通用快速模型 (u2net)": "u2net",
    "静物与通用主体 (isnet-general) - 推荐": "isnet-general-use",
    "动漫插画专属 (isnet-anime)": "isnet-anime",
    "真实人物全身 (u2net_human_seg)": "u2net_human_seg"
}

alpha_thresh = st.sidebar.slider(
    "抠图边缘保留度",
    min_value=10, max_value=200, value=60, step=10,
    help="数值越小，保留的边缘细节（如毛发、半透明部分）越多。"
)
use_clahe = st.sidebar.toggle("开启细节对比度强化", value=True, help="可增强相近颜色（如纯白物体）的立体感与轮廓。")

uploaded_file = st.file_uploader("请上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("原图预览")
    st.image(image, width=300)

    if st.button("为wyw生成拼豆图纸", type="primary"):
        st.info("💡 提示：正在处理图片，首次加载新 AI 模型可能需要几十秒，请稍候...")
        with st.spinner("正在生成中..."):

            fig, bead_counts = create_blueprint(
                image,
                grid_size,
                PERLER_COLORS,
                use_ai,
                model_dict[model_choice],
                use_clahe,
                alpha_thresh,
                brightness_val,
                use_dither
            )

            st.subheader("施工图纸")
            st.pyplot(fig)

            st.subheader("拼豆消耗清单 (真实可购型号)")
            if not bead_counts:
                st.error(
                    "🚨 **警告：完全没有识别到图片主体！** \n\nAI 可能把整张图片都当成背景误删了。\n\n👉 **解决办法**：请在左侧尝试**更换其他的 AI 抠图模型**，或者直接**关闭 AI 智能抠图**开关，然后重新生成！",
                    icon="🚨")
            else:
                # 【关键修改】使用单行拼接HTML以防止 Streamlit Markdown 引擎乱码，完美生成带边框小色块
                html_legend = "<div style='display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;'>"
                sorted_counts = sorted(bead_counts.items(), key=lambda x: x[1], reverse=True)

                for code, count in sorted_counts:
                    rgb = PERLER_COLORS[code]["rgb"]
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    text_color = "white" if luminance < 130 else "black"
                    
                    # 紧凑排列的单行 HTML 确保渲染无误
                    block_html = f"<div style='display:flex; border:1px solid #c0c0c0; border-radius:4px; overflow:hidden; font-family:sans-serif; font-size:13px; box-shadow:1px 1px 2px rgba(0,0,0,0.05);'><div style='background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); color:{text_color}; padding:4px 8px; font-weight:bold; border-right:1px solid #c0c0c0; text-align:center;'>{code}</div><div style='background-color:#ffffff; color:#000000; padding:4px 8px; text-align:center; min-width:30px;'>{count}</div></div>"
                    html_legend += block_html

                html_legend += "</div>"
                st.markdown(html_legend, unsafe_allow_html=True)

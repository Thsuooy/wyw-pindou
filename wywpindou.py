import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import cv2
from rembg import remove, new_session

# ================= 1. 官方色板 =================
PERLER_COLORS = {
    "W1": {"name": "白色", "rgb": (255, 255, 255)},
    "G1": {"name": "浅灰", "rgb": (185, 188, 190)},
    "G2": {"name": "中灰", "rgb": (137, 141, 144)},
    "G3": {"name": "深灰", "rgb": (84, 88, 90)},
    "B1": {"name": "黑色", "rgb": (0, 0, 0)},
    "R1": {"name": "大红", "rgb": (201, 36, 49)},
    "R2": {"name": "蔓越莓红", "rgb": (173, 50, 87)},
    "R3": {"name": "樱桃红", "rgb": (180, 40, 60)},
    "P1": {"name": "粉红", "rgb": (234, 113, 157)},
    "P2": {"name": "品红", "rgb": (202, 53, 116)},
    "O1": {"name": "橙色", "rgb": (240, 116, 45)},
    "O2": {"name": "奶酪橙", "rgb": (243, 152, 44)},
    "O3": {"name": "锈红", "rgb": (158, 66, 42)},
    "Y1": {"name": "明黄", "rgb": (240, 194, 21)},
    "Y2": {"name": "淡黄", "rgb": (245, 230, 120)},
    "S1": {"name": "肤色/桃粉", "rgb": (244, 180, 154)},
    "S2": {"name": "沙色", "rgb": (224, 196, 151)},
    "GR1": {"name": "正绿", "rgb": (55, 143, 85)},
    "GR2": {"name": "浅绿", "rgb": (86, 183, 121)},
    "GR3": {"name": "深绿", "rgb": (43, 94, 62)},
    "GR4": {"name": "薄荷绿", "rgb": (100, 200, 160)},
    "BL1": {"name": "正蓝", "rgb": (51, 105, 178)},
    "BL2": {"name": "浅蓝", "rgb": (72, 165, 219)},
    "BL3": {"name": "深蓝", "rgb": (41, 62, 137)},
    "BL4": {"name": "绿松石蓝", "rgb": (58, 168, 193)},
    "PU1": {"name": "正紫", "rgb": (111, 56, 137)},
    "PU2": {"name": "李子紫", "rgb": (142, 65, 143)},
    "PU3": {"name": "淡紫", "rgb": (180, 150, 200)},
    "BR1": {"name": "深棕", "rgb": (94, 62, 46)},
    "BR2": {"name": "浅棕", "rgb": (153, 102, 51)},
    "BR3": {"name": "黄褐色", "rgb": (204, 163, 114)},
    "BR4": {"name": "奶油糖", "rgb": (216, 145, 58)}
}

# ================= 2. 核心图像处理函数 =================
def apply_palette_and_dither(img_small, alpha_array, palette, use_dither, alpha_threshold):
    keys = list(palette.keys())
    flat_palette = []
    for k in keys:
        flat_palette.extend(palette[k]["rgb"])
    flat_palette.extend([0] * (768 - len(flat_palette)))

    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(flat_palette)

    img_rgb = img_small.convert("RGB")
    dither_mode = Image.FLOYDSTEINBERG if use_dither else Image.NONE

    img_quant = img_rgb.quantize(palette=pal_img, dither=dither_mode)
    quant_array = np.array(img_quant)

    h, w = quant_array.shape
    mapped_pixels = np.zeros((h, w, 3), dtype=np.uint8)
    code_matrix = np.empty((h, w), dtype=object)

    for y in range(h):
        for x in range(w):
            if alpha_array[y, x] < alpha_threshold:
                mapped_pixels[y, x] = [255, 255, 255]
                code_matrix[y, x] = None
            else:
                idx = quant_array[y, x]
                if idx < len(keys):
                    code = keys[idx]
                    mapped_pixels[y, x] = palette[code]["rgb"]
                    code_matrix[y, x] = code
                else:
                    mapped_pixels[y, x] = [255, 255, 255]
                    code_matrix[y, x] = None

    return mapped_pixels, code_matrix


def create_blueprint(image, max_size, palette, use_ai_bg_removal, ai_model_name, use_clahe, alpha_threshold, brightness, use_dither):
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
    ax.tick_params(top=True, labeltop=True, bottom=True, labelbottom=True, left=True, labelleft=True, right=True, labelright=True)

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
                font_s = 9 if max_size <= 40 else (6 if max_size <= 80 else 4)
                ax.text(x, y, code, ha='center', va='center', color=text_color, fontsize=font_s, fontweight='bold')

    plt.tight_layout()
    return fig, bead_counts


# ================= 3. 网页界面设计 =================
st.set_page_config(page_title="wyw专属拼豆图纸生成器", page_icon="🎨", layout="wide")
st.title("🎨 wyw专属拼豆图纸生成器")
st.write("提供全套色卡匹配、智能主体识别与边缘优化功能。")

st.sidebar.header("参数设置")
grid_size = st.sidebar.slider("图纸精细度 (最长边豆子数)", min_value=15, max_value=120, value=50, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("色彩校正 (解决颜色不准)")
st.sidebar.info("💡 **小贴士**：如果白色物体生成出来发黄、发灰或变成了棕色，请调高曝光亮度！")
brightness_val = st.sidebar.slider("曝光亮度补偿", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="1.0为原图。调高可消除阴影，让物体更白亮。")
use_dither = st.sidebar.toggle("开启像素抖动 (解决色彩断层)", value=True, help="当颜色缺失时，用两种相近颜色的豆子交替排列来欺骗视觉，效果更逼真！")

st.sidebar.markdown("---")
st.sidebar.subheader("AI 抠图与画质优化")
use_ai = st.sidebar.toggle("开启 AI 智能抠图", value=True)

model_choice = st.sidebar.selectbox(
    "选择 AI 抠图模型",
    ["静物与通用主体 (isnet-general) - 推荐", "通用快速模型 (u2net)", "动漫插画专属 (isnet-anime)", "真实人物全身 (u2net_human_seg)"],
    index=0
)
model_dict = {
    "通用快速模型 (u2net)": "u2net",
    "静物与通用主体 (isnet-general) - 推荐": "isnet-general-use",
    "动漫插画专属 (isnet-anime)": "isnet-anime",
    "真实人物全身 (u2net_human_seg)": "u2net_human_seg"
}

alpha_thresh = st.sidebar.slider("抠图边缘保留度", min_value=10, max_value=200, value=60, step=10, help="数值越小，保留的边缘细节越多。")
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
                image, grid_size, PERLER_COLORS, use_ai, model_dict[model_choice], use_clahe, alpha_thresh, brightness_val, use_dither
            )

            st.subheader("施工图纸")
            st.pyplot(fig)

            st.subheader("拼豆消耗清单")
            if not bead_counts:
                st.error("🚨 **警告：完全没有识别到图片主体！** \n\nAI 可能把整张图片都当成背景误删了。\n\n👉 **解决办法**：请在左侧尝试**更换其他的 AI 抠图模型**，或者直接**关闭 AI 智能抠图**开关，然后重新生成！", icon="🚨")
            else:
                # 【关键修复】取消了 HTML 的多行缩进，防止 Streamlit 将其解析为代码块
                html_legend = "<div style='display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px;'>"
                sorted_counts = sorted(bead_counts.items(), key=lambda x: x[1], reverse=True)

                for code, count in sorted_counts:
                    rgb = PERLER_COLORS[code]["rgb"]
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    text_color = "white" if luminance < 130 else "black"
                    
                    # 使用紧凑的单行 HTML 确保完美渲染成带边框的色块
                    block_html = f"<div style='display:flex; border:1px solid #c0c0c0; border-radius:4px; overflow:hidden; font-family:sans-serif; font-size:13px; box-shadow:1px 1px 2px rgba(0,0,0,0.05);'><div style='background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); color:{text_color}; padding:4px 8px; font-weight:bold; border-right:1px solid #c0c0c0; text-align:center;'>{code}</div><div style='background-color:#ffffff; color:#000000; padding:4px 8px; text-align:center; min-width:30px;'>{count}</div></div>"
                    html_legend += block_html

                html_legend += "</div>"
                
                # 渲染 HTML
                st.markdown(html_legend, unsafe_allow_html=True)

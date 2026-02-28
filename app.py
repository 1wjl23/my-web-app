import os
import sqlite3
import socket
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import os
import requests
import torch

# 配置区
MODEL_PATH = "model.pth" 
# ！！！请在下方替换为你刚复制的 model.pth 的 Release 链接
MODEL_URL = "https://github.com/1wjl23/my-web-app/releases/download/102/model.pth" 

# 自动下载函数
def load_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("首次运行，正在加载 AI 模型，请稍候..."):
            r = requests.get(MODEL_URL, stream=True)
            if r.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                st.success("模型同步完成！")
            else:
                st.error("下载失败，请检查链接是否正确")
                return None
    return MODEL_PATH

# 在你的主程序加载模型的地方改用这个
path = load_model_file()
if path:
    # 这里写你原本加载模型的代码，比如：
    # model = torch.load(path, map_location=torch.device('cpu'))
    st.write("✅ 模型已就绪")
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet34


# ===================== 基础配置（与“运行模型代码”保持一致） =====================
# 注意：这个字典索引顺序必须与训练时 `sorted(os.listdir(TRAIN_DIR))` 的类别顺序一致
CLASS_IDX_TO_NAME: Dict[int, str] = {
    0: "三七",
    1: "人参",
    2: "甘草",
    3: "白术",
    4: "白芍",
    5: "艾叶",
    6: "苍术",
    7: "茵陈",
    8: "附子",
    9: "黄芩",
}
CLASS_NUM = len(CLASS_IDX_TO_NAME)
MODEL_PATH = Path(__file__).parent / "中药分类模型_优化版.pth"

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = APP_DIR / "uploads"
DB_PATH = DATA_DIR / "herb_system.sqlite3"


DEFAULT_HERB_INFO: Dict[str, Dict[str, str]] = {
    "三七": {
        "产地": "云南、广西等地",
        "药用部位": "根",
        "功效": "散瘀止血，消肿定痛，化瘀通络，益气活血。用于咯血、吐血、衄血、便血、崩漏、外伤出血等各类出血证；亦可治胸腹刺痛、跌扑肿痛、瘀血痹阻、经络不通；善“止血不留瘀，化瘀不伤正”，为血证要药，现代亦常用于心脉瘀阻、胸闷心痛、气虚血瘀诸症。",
    },
    "人参": {
        "产地": "东北三省为主",
        "药用部位": "根及根茎",
        "功效": "大补元气，复脉固脱，益气摄血。用于体虚欲脱，肢冷脉微，气不摄血，崩漏下血；心力衰竭，心原性休克。",
    },
    "甘草": {
        "产地": "内蒙古、甘肃等地",
        "药用部位": "根及根茎",
        "功效": "补脾益气，清热解毒，祛痰止咳，缓急止痛，调和诸药。用于脾胃虚弱，倦怠乏力，心悸气短，咳嗽痰多，脘腹、四肢挛急疼痛，痈肿疮毒，缓解药物毒性、烈性。",
    },
    "白术": {
        "产地": "浙江、安徽等地",
        "药用部位": "根茎",
        "功效": "健脾益气，燥湿利水，止汗，安胎。用于脾虚食少，腹胀泄泻，痰饮眩悸，水肿，自汗，胎动不安。",
    },
    "白芍": {
        "产地": "浙江、安徽等地",
        "药用部位": "根",
        "功效": "养血调经，敛阴止汗，柔肝止痛，平抑肝阳。用于血虚萎黄，月经不调，自汗，盗汗，胁痛，腹痛，四肢挛痛，头痛眩晕。",
    },
    "艾叶": {
        "产地": "湖北、安徽等地",
        "药用部位": "叶",
        "功效": "温经止血，散寒止痛；外用祛湿止痒。用于吐血，衄血，崩漏，月经过多，胎漏下血，少腹冷痛，经寒不调，宫冷不孕；外治皮肤瘙痒。",
    },
    "苍术": {
        "产地": "江苏、湖北等地",
        "药用部位": "根茎",
        "功效": "燥湿健脾，祛风散寒，明目。用于湿阻中焦，脘腹胀满，泄泻，水肿，脚气痿蹙，风湿痹痛，风寒感冒，夜盲，眼目昏涩。",
    },
    "茵陈": {
        "产地": "陕西、山西等地",
        "药用部位": "地上部分",
        "功效": "清利湿热，利胆退黄。用于黄疸尿少，湿温暑湿，胸闷呕恶，湿热黄疸，胆胀胁痛；传染性肝炎。",
    },
    "附子": {
        "产地": "四川、陕西等地",
        "药用部位": "子根加工品",
        "功效": "回阳救逆，补火助阳，散寒止痛。用于亡阳虚脱，肢冷脉微，心阳不足，胸痹心痛，虚寒吐泻，脘腹冷痛，肾阳虚衰，阳痿宫冷，阴寒水肿，阳虚外感，寒湿痹痛。",
    },
    "黄芩": {
        "产地": "河北、山西等地",
        "药用部位": "根",
        "功效": "清热燥湿，泻火解毒，止血，安胎。用于湿温、暑湿，胸闷呕恶，湿热痞满，泻痢，黄疸，肺热咳嗽，高热烦渴，血热吐衄，痈肿疮毒，胎动不安。",
    },
}


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS herbs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL UNIQUE,
              origin TEXT NOT NULL,
              part_used TEXT NOT NULL,
              effect TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              herb_name TEXT NOT NULL,
              confidence REAL NOT NULL,
              image_path TEXT NOT NULL
            );
            """
        )
        conn.commit()

    seed_default_herbs_if_empty()


def seed_default_herbs_if_empty() -> None:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM herbs;").fetchone()
        cnt = int(row["cnt"]) if row else 0
        if cnt > 0:
            return

        for name, info in DEFAULT_HERB_INFO.items():
            conn.execute(
                """
                INSERT INTO herbs(name, origin, part_used, effect, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (name, info.get("产地", "无"), info.get("药用部位", "无"), info.get("功效", "无"), _now_str(), _now_str()),
            )
        conn.commit()


def upsert_herb(name: str, origin: str, part_used: str, effect: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO herbs(name, origin, part_used, effect, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              origin=excluded.origin,
              part_used=excluded.part_used,
              effect=excluded.effect,
              updated_at=excluded.updated_at;
            """,
            (name, origin, part_used, effect, _now_str(), _now_str()),
        )
        conn.commit()


def get_herb_by_name(name: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT name, origin, part_used, effect, updated_at FROM herbs WHERE name=?;",
            (name,),
        ).fetchone()
    return dict(row) if row else None


def list_herbs_df() -> pd.DataFrame:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT name AS 药材名称, origin AS 产地, part_used AS 药用部位, effect AS 功效, updated_at AS 更新时间 FROM herbs ORDER BY name;"
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def add_record(created_at: str, herb_name: str, confidence: float, image_path: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO recognition_records(created_at, herb_name, confidence, image_path)
            VALUES (?, ?, ?, ?);
            """,
            (created_at, herb_name, confidence, image_path),
        )
        conn.commit()


def list_records() -> list[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, herb_name, confidence, image_path
            FROM recognition_records
            ORDER BY id DESC;
            """
        ).fetchall()
    return [dict(r) for r in rows]


def delete_record(record_id: int) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM recognition_records WHERE id=?;", (record_id,))
        conn.commit()


# ===================== 模型加载与预处理（与“运行模型代码”保持一致） =====================
@st.cache_resource
def load_model_bundle() -> Tuple[torch.nn.Module, transforms.Compose, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, CLASS_NUM)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, device


def predict_image(img: Image.Image) -> Tuple[Optional[str], Optional[float]]:
    model, transform, device = load_model_bundle()
    if not MODEL_PATH.exists():
        return None, None

    try:
        img = img.convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred_idx = torch.max(outputs, 1)
            pred_name = CLASS_IDX_TO_NAME.get(int(pred_idx.item()), "未知")
            conf = torch.softmax(outputs, 1)[0][pred_idx].item() * 100.0
        return pred_name, float(conf)
    except Exception:
        return None, None


@dataclass(frozen=True)
class SavedUpload:
    pil_image: Image.Image
    saved_path: Path


def save_upload_to_disk(upload) -> Optional[SavedUpload]:
    if upload is None:
        return None

    _ensure_dirs()
    suffix = Path(upload.name).suffix.lower() if upload.name else ".png"
    if suffix not in [".jpg", ".jpeg", ".png"]:
        suffix = ".png"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = Path(upload.name).stem if upload.name else "upload"
    safe_stem = "".join(ch for ch in safe_stem if ch.isalnum() or ch in ["-", "_"])[:40] or "upload"
    filename = f"{ts}_{safe_stem}{suffix}"
    out_path = UPLOAD_DIR / filename

    data = upload.getvalue()
    out_path.write_bytes(data)
    pil_img = Image.open(out_path).convert("RGB")
    return SavedUpload(pil_image=pil_img, saved_path=out_path)


# ===================== Streamlit UI =====================
def set_style() -> None:
    st.set_page_config(page_title="中药材智能识别与管理系统", page_icon="🌿", layout="wide")
    st.markdown(
        """
        <style>
          /* 避免顶部工具栏遮挡标题：不要把 padding-top 调得太小 */
          .block-container { padding-top: 3.25rem; padding-bottom: 2rem; }
          [data-testid="stSidebar"] { border-right: 1px solid rgba(49,51,63,0.12); }
          .metric-card {
            border: 1px solid rgba(49,51,63,0.12);
            border-radius: 14px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.75);
          }
          .small-muted { color: rgba(49,51,63,0.65); font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_recognize() -> None:
    st.title("中药材识别")
    if not MODEL_PATH.exists():
        st.warning(f"未找到模型文件：`{MODEL_PATH.name}`。请把模型放在 `app.py` 同目录后刷新页面。")

    st.write("支持点击选择或拖放上传图片（JPG/PNG）。识别成功后会自动写入识别记录。")

    upload = st.file_uploader("选择文件 / 拖放上传", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    saved = save_upload_to_disk(upload)

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("#### 图片预览")
        if saved is None:
            st.info("请先上传一张药材图片。")
        else:
            st.image(saved.pil_image, use_container_width=True)
            st.caption(f"已保存：`{saved.saved_path.name}`")

    with right:
        st.markdown("#### 识别结果")
        if saved is None:
            st.markdown('<div class="metric-card"><div class="small-muted">等待识别…</div></div>', unsafe_allow_html=True)
            return

        with st.spinner("正在识别..."):
            pred_name, conf = predict_image(saved.pil_image)

        if pred_name is None or conf is None:
            st.error("识别失败（模型未加载或推理异常）。")
            return

        herb = get_herb_by_name(pred_name) or {}
        conf_text = f"{conf:.2f}%"

        st.markdown(
            f"""
            <div class="metric-card">
              <div style="font-size: 1.05rem;"><b>药材名称</b>：{pred_name}</div>
              <div style="margin-top: 6px;"><b>置信度</b>：{conf_text}</div>
              <hr style="margin: 12px 0; border: none; border-top: 1px solid rgba(49,51,63,0.12);" />
              <div><b>产地</b>：{herb.get("origin", "无")}</div>
              <div style="margin-top: 6px;"><b>药用部位</b>：{herb.get("part_used", "无")}</div>
              <div style="margin-top: 6px;"><b>功效</b>：{herb.get("effect", "无")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        add_record(
            created_at=_now_str(),
            herb_name=pred_name,
            confidence=float(conf),
            image_path=str(saved.saved_path.relative_to(APP_DIR)),
        )
        st.success("已写入识别记录。")


def page_herb_manage() -> None:
    st.title("药材信息管理")
    st.write("这里维护药材百科信息（产地 / 药用部位 / 功效）。识别页面会自动从数据库匹配显示。")

    df = list_herbs_df()
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    tab_add, tab_edit = st.tabs(["新增", "编辑"])

    with tab_add:
        with st.form("add_form", clear_on_submit=True):
            name = st.text_input("药材名称（唯一）", placeholder="例如：三七")
            origin = st.text_input("产地", placeholder="例如：云南、广西等地")
            part_used = st.text_input("药用部位", placeholder="例如：根")
            effect = st.text_area("功效", height=120, placeholder="例如：散瘀止血，消肿定痛…")
            submitted = st.form_submit_button("新增 / 覆盖保存", type="primary")
        if submitted:
            if not name.strip():
                st.error("药材名称不能为空。")
            else:
                upsert_herb(name.strip(), origin.strip() or "无", part_used.strip() or "无", effect.strip() or "无")
                st.success("已保存。请在左侧刷新或切换页面查看更新。")

    with tab_edit:
        names = df["药材名称"].tolist() if not df.empty else []
        if not names:
            st.info("当前数据库还没有药材信息。请先在“新增”中添加。")
        else:
            selected = st.selectbox("选择要编辑的药材", options=names)
            herb = get_herb_by_name(selected) or {}
            with st.form("edit_form"):
                origin = st.text_input("产地", value=str(herb.get("origin", "无")))
                part_used = st.text_input("药用部位", value=str(herb.get("part_used", "无")))
                effect = st.text_area("功效", value=str(herb.get("effect", "无")), height=150)
                saved = st.form_submit_button("保存修改", type="primary")
            if saved:
                upsert_herb(selected, origin.strip() or "无", part_used.strip() or "无", effect.strip() or "无")
                st.success("已更新。")


def page_records_manage() -> None:
    st.title("识别记录管理")
    records = list_records()
    if not records:
        st.info("暂无识别记录。")
        return

    st.write("点击对应行右侧“删除”可一键删除该条记录。")
    for rec in records[:200]:
        c1, c2, c3, c4, c5 = st.columns([0.16, 0.14, 0.18, 0.12, 0.12], vertical_alignment="center")
        c1.write(rec["created_at"])
        c2.write(rec["herb_name"])
        c3.write(f'{rec["confidence"]:.2f}%')
        c4.write(rec["image_path"])
        if c5.button("删除", key=f"del_{rec['id']}", type="secondary"):
            delete_record(int(rec["id"]))
            st.success(f"已删除记录 #{rec['id']}")
            st.rerun()


def main() -> None:
    set_style()
    init_db()

    with st.sidebar:
        st.title("🌿 中药材系统")
        page = st.radio(
            "导航",
            ["中药材识别", "药材信息管理", "识别记录管理"],
            index=0,
        )
        st.caption("本系统使用本地 SQLite 存储百科与识别记录。")

    if page == "中药材识别":
        page_recognize()
    elif page == "药材信息管理":
        page_herb_manage()
    else:
        page_records_manage()


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _bootstrap_if_bare_python() -> None:
    if _is_running_in_streamlit():
        return

    def find_free_port(preferred: int = 8501, tries: int = 50) -> int:
        for p in range(preferred, preferred + tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    # Windows 下 SO_REUSEADDR 可能导致“误判可用”，这里严格用 0.0.0.0 检测
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        return preferred

    env_port = os.environ.get("STREAMLIT_PORT")
    preferred_port = int(env_port) if (env_port and env_port.isdigit()) else 8501
    port = find_free_port(preferred_port)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.port",
        str(port),
    ]
    subprocess.Popen(cmd, cwd=str(APP_DIR))
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    raise SystemExit(0)


if __name__ == "__main__":
    _bootstrap_if_bare_python()
    main()


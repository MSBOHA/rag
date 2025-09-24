import streamlit as st
import os
import yaml
import subprocess
from pathlib import Path
import sys
from typing import Any

st.set_page_config(page_title="RAG WebUI", layout="wide")

# 加载配置
def get_config_path():
    # 自动定位到项目根目录下的 configs/config.yaml
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "config.yaml"
    return str(config_path)

def load_config(config_path=None):
    if config_path is None:
        config_path = get_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config, config_path=None):
    if config_path is None:
        config_path = get_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

# 页面布局
st.title("🧠 RAG Pipeline WebUI")
tabs = st.tabs(["配置", "构建索引", "检索问答", "对话历史"])

with tabs[0]:
    st.header("配置参数")
    config = load_config()
    config_types = {}

    def render_input(label: str, value: Any):
        config_types[label] = type(value)
        if isinstance(value, bool):
            return st.checkbox(label, value=value)
        elif isinstance(value, int):
            return st.number_input(label, value=value, step=1, format="%d")
        elif isinstance(value, float):
            return st.number_input(label, value=value, format="%f")
        else:
            return st.text_input(label, str(value))

    for key, value in config.items():
        if isinstance(value, dict):
            with st.expander(key, expanded=False):
                for k, v in value.items():
                    new_v = render_input(f"{key}.{k}", v)
                    config[key][k] = new_v
        else:
            config[key] = render_input(key, value)

    def cast_value(val, typ):
        if typ is bool:
            if isinstance(val, bool):
                return val
            return str(val).lower() in ("true", "1", "yes", "on")
        if typ is int:
            try:
                return int(val)
            except Exception:
                return 0
        if typ is float:
            try:
                return float(val)
            except Exception:
                return 0.0
        return val

    if st.button("保存配置"):
        # 类型还原
        for key, value in config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    typ = config_types.get(f"{key}.{k}", str)
                    config[key][k] = cast_value(v, typ)
            else:
                typ = config_types.get(key, str)
                config[key] = cast_value(value, typ)
        save_config(config)
        st.success("配置已保存！")

with tabs[1]:
    st.header("批量构建索引")
    doc_folder = st.text_input("文档主目录", config.get("doc_path", "./docs"))
    if st.button("开始批量构建"):
        if not doc_folder or not os.path.isdir(doc_folder):
            st.error(f"文档主目录无效：{doc_folder}")
        else:
            with st.spinner("正在批量构建索引..."):
                config_path = get_config_path()
                cmd = [
                    sys.executable, "-u", str(Path(__file__).parent.parent / "scripts" / "build_index.py"), "--config", config_path, "--doc_path", doc_folder
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent, timeout=300)
                if result.returncode == 0:
                    st.success("索引构建完成！")
                    if result.stdout:
                        st.text(result.stdout)
                else:
                    st.error(f"构建失败: {result.stderr}")
                    if result.stdout:
                        st.text(result.stdout)

with tabs[2]:
    st.header("检索与问答（支持多轮对话）")
    # 支持多库选择
    index_root = Path(__file__).parent.parent / "indexes"
    if index_root.exists():
        dbs = [f.name for f in index_root.iterdir() if f.is_dir()]
    else:
        dbs = []
    db_selected = st.multiselect("选择向量库（可多选）", dbs, default=dbs[:1])
    user_input = st.text_area("请输入问题", "")
    chat_mode = st.checkbox("多轮对话模式", value=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if st.button("发送") and user_input:
        # 组装命令
        config_path = get_config_path()
        cmd = [sys.executable, "-u", str(Path(__file__).parent.parent / "scripts" / "query.py"), "--config", config_path, "--question", user_input]
        if chat_mode:
            cmd.append("--chat")
        if db_selected:
            for db in db_selected:
                cmd.extend(["--db", db])
        # 维护多轮对话历史
        if chat_mode:
            for msg in st.session_state.chat_history:
                cmd.extend(["--history", msg["role"]+":"+msg["content"]])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent, timeout=300)
        if result.returncode == 0:
            answer = result.stdout.strip()
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.success(answer)
            if result.stdout:
                st.text(result.stdout)
        else:
            st.error(f"查询失败: {result.stderr}")
            if result.stdout:
                st.text(result.stdout)
    # 展示对话历史
    for msg in st.session_state.chat_history[-10:]:
        if msg["role"] == "user":
            st.markdown(f"**用户：** {msg['content']}")
        else:
            st.markdown(f"**助手：** {msg['content']}")

with tabs[3]:
    st.header("对话历史")
    for msg in st.session_state.get("chat_history", []):
        st.write(f"[{msg['role']}] {msg['content']}")

import os
import json
import time
import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv, find_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# ===========================
# Инициализация страницы
# ===========================
load_dotenv(find_dotenv())

st.set_page_config(page_title="Тест изображения", page_icon="🧪", layout="wide")
st.title("🧪 Тест изображения")

# ===========================
# Хелперы
# ===========================
def load_df_from_state_or_file() -> pd.DataFrame:
    df = st.session_state.get("df")
    if df is not None and not df.empty:
        return df
    # fallback
    return pd.read_csv("data.csv")

def validate_df(df: pd.DataFrame):
    required = ["image_name", "question", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"В CSV отсутствуют обязательные столбцы: {', '.join(missing)}")
        st.stop()

@st.cache_data(show_spinner=False)
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def image_mime(path: Path) -> str:
    suf = path.suffix.lower().lstrip(".")
    return f"image/{'jpeg' if suf in ['jpg', 'jpeg'] else suf}"

def get_llm(api_key: str, model_name: str, temperature: float = 0.0, max_tokens: int = 2048) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        default_headers={"X-Title": "Vision Model Evaluator - Test"},
    )

def readable_time(sec: float) -> str:
    if sec < 1: return f"{sec*1000:.0f} мс"
    if sec < 60: return f"{sec:.1f} с"
    m = int(sec // 60); s = int(sec % 60)
    return f"{m}м {s}с"

def create_html_for_detailed_qa_pair(index: int, question: str, gen_answer: str, true_answer: str, time_elapsed: float) -> str:
    """
    creates well formatted colored html block for detailed Q&A pair
    :param index:
    :param question:
    :param gen_answer:
    :param true_answer:
    :param time_elapsed:
    :return:
    """
    return f"""
    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <h4 style="margin-top: 0;">Вопрос {index}</h4>
        <p><b>Вопрос:</b> {question}</p>
        <p><b>Ответ модели:</b></p>
        <div style="background-color: #f9f9f9; padding: 12px; border-radius: 4px; white-space: pre-wrap;">{gen_answer}</div>
        <p><b>Верный ответ:</b></p>
        <div style="background-color: #e8f5e9; padding: 12px; border-radius: 4px; white-space: pre-wrap;">{true_answer}</div>
        <p style="font-size: 0.9em; color: #555;">Время: {readable_time(time_elapsed)}</p>
    </div>
    """

# ===========================
# Сайдбар: настройки
# ===========================
with st.sidebar:
    st.header("⚙️ Настройки")

    # Путь к папке с изображениями — используем то, что было на основной странице, если есть
    default_img_dir = st.session_state.get("image_folder_path", "images")
    image_folder_path = st.text_input("Папка с изображениями", value=default_img_dir)
    st.session_state.image_folder_path = image_folder_path  # синхронизируем в сессию

    st.divider()
    st.caption("Подключение к OpenRouter")
    openrouter_key = st.text_input(
        "API-ключ OpenRouter",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", ""),
        help="Ключ не сохраняется в приложении."
    )

    st.markdown("Модель")
    model_name = st.text_input(
        "Имя модели",
        value="openai/gpt-4o-mini",
        help="Например: openai/gpt-4o, openai/gpt-4o-mini"
    )
    temperature = st.slider("Температура", 0.0, 1.0, 0.0, 0.05)
    max_tokens = st.slider("Max tokens", 512, 8192, 2048, 128)

# ===========================
# Данные
# ===========================
try:
    df = load_df_from_state_or_file()
    validate_df(df)
except Exception as e:
    st.error(f"Ошибка загрузки датасета: {e}")
    st.stop()

if df.empty:
    st.warning("Датасет пуст. Загрузите CSV на главной странице или положите файл data.csv.")
    st.stop()

# ===========================
# Выбор изображения и вопросов
# ===========================
unique_images = sorted(df["image_name"].astype(str).unique().tolist())
if not unique_images:
    st.warning("В датасете нет изображений.")
    st.stop()

col_top_left, col_top_right = st.columns([1, 1])
with col_top_left:
    selected_image = st.selectbox("Выберите изображение из датасета", options=unique_images, index=0)

rows_for_image = df[df["image_name"].astype(str) == str(selected_image)].copy()
available_q = len(rows_for_image)

with col_top_right:
    st.metric("Доступно вопросов по изображению", available_q)

if available_q == 0:
    st.warning("Для выбранного изображения нет вопросов.")
    st.stop()

# Контролы выбора 3–5 вопросов (адаптируемся под доступное количество)
min_q = 3 if available_q >= 3 else 1
max_q = min(5, available_q)
selection_col1, selection_col2, selection_col3 = st.columns([1, 1, 1])
with selection_col1:
    random_sample = st.toggle("Случайный выбор вопросов", value=True)
with selection_col2:
    num_questions = st.slider("Количество вопросов", min_value=min_q, max_value=max_q, value=max_q, step=1)
with selection_col3:
    seed = st.number_input("Seed", min_value=0, value=42, step=1)

# Предпросмотр изображения и выбранных вопросов
prev_left, prev_right = st.columns([1.1, 1.3])
with prev_left:
    img_path = Path(image_folder_path) / str(selected_image)
    if img_path.exists():
        try:
            st.image(Image.open(img_path), caption=str(selected_image))
        except Exception:
            st.warning("Не удалось отобразить изображение.")
    else:
        st.error(f"Файл не найден: {img_path}")

with prev_right:
    st.markdown("Предпросмотр вопросов")
    if random_sample:
        preview_df = rows_for_image.sample(n=num_questions, random_state=seed)
    else:
        preview_df = rows_for_image.head(n=num_questions)
    # st.dataframe(preview_df[["question", "answer"]].reset_index(drop=True), use_container_width=True)
    html = ""
    for i, row in preview_df.iterrows():
        q = str(row["question"])
        a = str(row["answer"])
        # small font, answer in expander
        html += f"""
        <div style="margin-bottom: 12px;">
            <b>Вопрос {i+1}:</b> {q}<br/>
            <details>
                <summary style="cursor: pointer; color: #0066cc;">Показать ответ</summary>
                <div style="margin-top: 4px; font-size: 0.9em; color: #333;">{a}</div>
            </details>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)


st.divider()

# ===========================
# Запуск теста
# ===========================
run_col, info_col = st.columns([1, 2])
with run_col:
    run = st.button("🚀 Запустить тест", use_container_width=True)
with info_col:
    st.caption("Будут прогнаны выбранные вопросы по одному изображению. Рекомендовано для быстрого теста модели.")

if run:
    if not openrouter_key:
        st.error("Укажите API-ключ OpenRouter.")
        st.stop()
    if not img_path.exists():
        st.error(f"Файл изображения не найден: {img_path}")
        st.stop()

    # Подготовка
    vision_llm = get_llm(openrouter_key, model_name, temperature=temperature, max_tokens=max_tokens)
    if random_sample:
        test_df = rows_for_image.sample(n=num_questions, random_state=seed)
    else:
        test_df = rows_for_image.head(n=num_questions)

    results = []
    progress = st.progress(0)
    status = st.empty()

    # Кодируем изображение один раз
    try:
        encoded = encode_image(img_path)
        mime = image_mime(img_path)
        img_url = f"data:{mime};base64,{encoded}"
    except Exception as e:
        st.error(f"Ошибка кодирования изображения: {e}")
        st.stop()

    start_all = time.time()
    for i, (_, r) in enumerate(test_df.iterrows(), start=1):
        q = str(r["question"])
        true_a = str(r["answer"])

        status.info(f"[{i}/{num_questions}] Вопрос: {q[:60]}{'...' if len(q)>60 else ''}")
        t0 = time.time()

        try:
            msg = HumanMessage(content=[
                {"type": "text", "text": q},
                {"type": "image_url", "image_url": {"url": img_url}}
            ])
            res = vision_llm.invoke([msg])
            gen_a = res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            gen_a = f"Ошибка генерации: {e}"

        elapsed = time.time() - t0

        results.append({
            "question": q,
            "generated_answer": gen_a,
            "true_answer": true_a,
            "time_elapsed_sec": elapsed
        })
        progress.progress(i / num_questions)

    total_time = time.time() - start_all
    status.success(f"Готово за {readable_time(total_time)}")
    progress.empty()

    # Табличка результатов
    st.subheader("Результаты")
    res_df = pd.DataFrame(results)
    # Переупорядочим столбцы
    res_df = res_df[["question", "generated_answer", "true_answer", "time_elapsed_sec"]]
    st.dataframe(res_df, use_container_width=True)

    # Детализация по вопросам
    with st.expander("Подробно по каждому вопросу", expanded=False):
        for idx, row in res_df.iterrows():
            html = create_html_for_detailed_qa_pair(
                index=idx + 1,
                question=row["question"],
                gen_answer=row["generated_answer"],
                true_answer=row["true_answer"],
                time_elapsed=row["time_elapsed_sec"]
            )
            st.markdown(html, unsafe_allow_html=True)

    # Скачивание результатов
    st.subheader("Скачать")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_")
    out_meta = {
        "type": "single_image_test",
        "image_name": str(selected_image),
        "model_name": model_name,
        "timestamp": timestamp,
        "total_items": len(results),
        "total_time_sec": total_time,
        "image_path": str(img_path)
    }
    out_payload = {"meta": out_meta, "results": results}

    json_bytes = json.dumps(out_payload, ensure_ascii=False, indent=2).encode("utf-8")
    csv_bytes = res_df.to_csv(index=False).encode("utf-8")

    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button(
            "⬇️ Скачать JSON",
            data=json_bytes,
            file_name=f"test_{safe_model}_{timestamp}.json",
            mime="application/json",
            use_container_width=True
        )
    with cdl2:
        st.download_button(
            "⬇️ Скачать CSV",
            data=csv_bytes,
            file_name=f"test_{safe_model}_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
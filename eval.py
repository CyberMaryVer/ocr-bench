import os
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv, find_dotenv

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

load_dotenv(find_dotenv())

# Page Configuration
st.set_page_config(
    page_title="OCR Image Bench",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main {
        padding: 2rem;
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Headers */
    h1 {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    h2 {
        color: #2d3748;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    h3 {
        color: #4a5568;
        font-weight: 600;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
    }

    /* Cards */
    .stExpander {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }

    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Success/Error/Warning/Info messages */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }

    /* File uploader */
    .uploadedFile {
        border: 2px dashed #cbd5e0;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #f7fafc;
    }

    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
        border-bottom: 2px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: transparent;
        border: none;
        color: #718096;
        font-weight: 500;
        font-size: 1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #667eea;
        border-bottom: 3px solid #667eea;
    }

    /* Dataframe */
    .dataframe {
        border: none !important;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Custom containers */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }

    /* Comparison table highlight */
    .comparison-highlight {
        background: linear-gradient(120deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-left: 4px solid #667eea;
        padding: 0.5rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Default Judge Prompt
DEFAULT_JUDGE_PROMPT = """
You are an impartial AI judge. Your task is to evaluate a generated answer against a ground truth answer for a given question about an image.
Your response MUST be a valid JSON object and nothing else. Do not include any text before or after the JSON object.

**Scoring Rubric:**
- **5: Perfect Match.** The generated answer is fully correct, complete, and aligns perfectly with the ground truth.
- **4: Mostly Correct.** The generated answer is substantially correct but may have minor inaccuracies or omissions.
- **3: Partially Correct.** The generated answer contains significant correct elements but also has notable errors or is incomplete.
- **2: Mostly Incorrect.** The generated answer is largely incorrect, with only small elements of truth.
- **1: Completely Incorrect.** The generated answer is completely wrong or irrelevant.

**Input:**
- **Question:** {question}
- **Ground Truth Answer:** {true_answer}
- **Generated Answer:** {generated_answer}

**Your Task:**
Analyze the 'Generated Answer' based on the 'Ground Truth Answer' and the rubric. Then, provide your response ONLY in the following JSON format.

**JSON OUTPUT EXAMPLE:**
```json
{{
  "score": 4,
  "justification": "The model correctly identified the main subject but missed a minor detail mentioned in the ground truth."
}}
"""


# --- Pydantic Models for Structured Output ---
class EvaluationResult(BaseModel):
    """The structured evaluation of the model's answer."""
    score: Literal[1, 2, 3, 4, 5] = Field(
        description="A score from 1 to 5, where 1 is completely wrong and 5 is perfectly correct and helpful."
    )
    justification: str = Field(
        description="A brief justification for the score, explaining why the answer was rated as such."
    )


# --- Helper Functions ---
def encode_image(image_path: Path) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_llm(api_key: str, model_name: str) -> ChatOpenAI:
    """Initializes and returns a ChatOpenAI instance for OpenRouter."""
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        max_tokens=2048,
        temperature=0.0,
        default_headers={
            "X-Title": "Vision Model Evaluator",
        }
    )


def load_all_benchmarks(folder="benchmarks"):
    """Load all benchmark results from the folder"""
    benchmark_files = list(Path(folder).glob("*.json"))
    all_data = []
    for file in sorted(benchmark_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(file, "r", encoding='utf-8') as f:
                data = json.load(f)
                data['meta']['filepath'] = str(file)
                data['meta']['filename'] = file.name
                all_data.append(data)
        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")
    return all_data


def create_score_distribution_chart(results_list):
    """Create a professional score distribution chart"""
    if not results_list:
        return None

    scores = [r['score'] for r in results_list]
    score_counts = pd.Series(scores).value_counts().sort_index()

    fig = go.Figure(data=[
        go.Bar(
            x=score_counts.index,
            y=score_counts.values,
            marker=dict(
                color=score_counts.index,
                colorscale='Viridis',
                showscale=False,
                line=dict(color='white', width=2)
            ),
            text=score_counts.values,
            textposition='outside',
            textfont=dict(size=14, color='#2d3748'),
            hovertemplate='Оценка: %{x}<br>Количество: %{y}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=dict(
            text='Распределение оценок',
            font=dict(size=20, color='#2d3748')
        ),
        xaxis=dict(
            title='Оценка',
            tickmode='linear',
            tick0=1,
            dtick=1,
            showgrid=False
        ),
        yaxis=dict(
            title='Количество ответов',
            showgrid=True,
            gridcolor='#e2e8f0'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        hovermode='x unified'
    )

    return fig


def create_performance_gauge(avg_score):
    """Create a professional gauge chart for average score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Средний балл", 'font': {'size': 24, 'color': '#2d3748'}},
        delta={'reference': 3, 'increasing': {'color': "#48bb78"}},
        gauge={
            'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "#cbd5e0"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 1], 'color': '#fc8181'},
                {'range': [1, 2], 'color': '#f6ad55'},
                {'range': [2, 3], 'color': '#f6e05e'},
                {'range': [3, 4], 'color': '#68d391'},
                {'range': [4, 5], 'color': '#48bb78'}
            ],
            'threshold': {
                'line': {'color': "#2d3748", 'width': 4},
                'thickness': 0.75,
                'value': avg_score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2d3748", 'family': "Inter"},
        height=300
    )

    return fig


def create_comparison_chart(comparison_data, include_time=True):
    """Create a comparison chart for multiple models"""
    df = pd.DataFrame(comparison_data)

    if include_time and 'avg_time' in df.columns:
        # Create subplot with two y-axes
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Средний балл', 'Среднее время (сек)'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Score bar chart
        fig.add_trace(
            go.Bar(
                x=df['model'],
                y=df['avg_score'],
                name='Средний балл',
                marker=dict(
                    color=df['avg_score'],
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(color='white', width=2)
                ),
                text=[f"{s:.2f}" for s in df['avg_score']],
                textposition='outside',
                hovertemplate='Модель: %{x}<br>Средний балл: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Time bar chart
        fig.add_trace(
            go.Bar(
                x=df['model'],
                y=df['avg_time'],
                name='Среднее время',
                marker=dict(
                    color=df['avg_time'],
                    colorscale='Plasma',
                    showscale=False,
                    line=dict(color='white', width=2)
                ),
                text=[f"{t:.2f}" for t in df['avg_time']],
                textposition='outside',
                hovertemplate='Модель: %{x}<br>Время: %{y:.2f} сек<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Сравнение моделей",
            title_font_size=20
        )
    else:
        # Single score chart
        fig = go.Figure(data=[
            go.Bar(
                x=df['model'],
                y=df['avg_score'],
                marker=dict(
                    color=df['avg_score'],
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(color='white', width=2)
                ),
                text=[f"{s:.2f}" for s in df['avg_score']],
                textposition='outside',
                hovertemplate='Модель: %{x}<br>Средний балл: %{y:.2f}<extra></extra>'
            )
        ])

        fig.update_layout(
            title='Сравнение моделей по среднему баллу',
            xaxis_title='Модель',
            yaxis_title='Средний балл',
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )

    return fig


# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'selected_result' not in st.session_state:
    st.session_state.selected_result = None

# Header with gradient
st.markdown("<h1>🎯 OCR Image Bench</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #718096; font-size: 1.2rem; margin-bottom: 2rem;'>Профессиональная оценка визуальных языковых моделей</p>",
    unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Конфигурация")
    st.markdown("---")

    # Dataset Configuration
    st.markdown("### 📁 Данные")

    image_folder_path = st.text_input(
        "Папка с изображениями",
        value="images",
        help="Путь к папке с изображениями для тестирования"
    )

    uploaded_file = st.file_uploader(
        "Загрузить CSV",
        type=["csv"],
        help="CSV файл должен содержать колонки: image_name, question, answer"
    )

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"✅ Загружено {len(st.session_state.df)} вопросов")

            # Show preview
            with st.expander("👁️ Предпросмотр данных", expanded=False):
                st.dataframe(st.session_state.df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Ошибка чтения CSV: {e}")
            st.session_state.df = None
    else:
        try:
            st.session_state.df = pd.read_csv("data.csv")
            st.info("ℹ️ Используется файл data.csv по умолчанию")
        except:
            st.warning("⚠️ Загрузите CSV файл для начала работы")

    # Sampling Configuration
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 🎲 Выборка данных")

        total_questions = len(st.session_state.df)

        col1, col2 = st.columns([3, 1])
        with col1:
            num_to_sample = st.slider(
                "Количество вопросов",
                min_value=1,
                max_value=total_questions,
                value=min(10, total_questions),
                help="Выберите подмножество вопросов для быстрой оценки"
            )
        with col2:
            st.metric("Всего", total_questions)

        if num_to_sample < total_questions:
            st.info(f"Будет использовано {num_to_sample} из {total_questions} вопросов")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["🚀 Запуск оценки", "📊 Результаты", "📈 Сравнение"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Настройки моделей")

        # API Configuration
        with st.expander("🔑 API Конфигурация", expanded=True):
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Ваш ключ не сохраняется",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                placeholder="sk-or-..."
            )

            col_api1, col_api2 = st.columns(2)
            with col_api1:
                model_name = st.text_input(
                    "🔮 Тестируемая модель",
                    value="openai/gpt-4o-mini",
                    help="Например: openai/gpt-4o, google/gemini-pro-vision",
                    placeholder="openai/gpt-4o-mini"
                )

            with col_api2:
                judge_name = st.text_input(
                    "⚖️ Модель-судья",
                    value="openai/gpt-4o-mini",
                    help="Рекомендуется мощная модель для оценки",
                    placeholder="openai/gpt-4o"
                )

            provider_tag = st.text_input(
                "🏷️ Тег провайдера",
                value="OpenRouter",
                help="Например: 'OpenAI', 'Anthropic', 'Google'",
                placeholder="OpenRouter"
            )

        # Judge Prompt Configuration
        with st.expander("📝 Промпт судьи", expanded=False):
            judge_prompt_template = st.text_area(
                "Шаблон промпта",
                value=DEFAULT_JUDGE_PROMPT,
                height=300,
                help="Используйте {question}, {true_answer}, {generated_answer} как плейсхолдеры",
                label_visibility="collapsed"
            )

    with col2:
        st.markdown("### Статус")

        # Status Card
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)

        if st.session_state.df is not None:
            st.success(f"✅ Данные загружены")
            st.metric("Вопросов к оценке", num_to_sample if 'num_to_sample' in locals() else len(st.session_state.df))
        else:
            st.warning("⚠️ Загрузите данные")
            st.metric("Вопросов к оценке", 0)

        if openrouter_key:
            st.success("✅ API ключ установлен")
        else:
            st.warning("⚠️ Введите API ключ")

        if Path(image_folder_path).is_dir():
            st.success(f"✅ Папка найдена")
        else:
            st.error(f"❌ Папка не найдена")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Run Button
        run_button = st.button(
            "Запустить оценку",
            use_container_width=True,
            disabled=st.session_state.is_running,
            type="primary"
        )

    # Run Evaluation
    if run_button:
        # Validation
        if not openrouter_key:
            st.error("❌ Пожалуйста, введите OpenRouter API ключ")
        elif st.session_state.df is None:
            st.error("❌ Пожалуйста, загрузите CSV файл")
        elif not Path(image_folder_path).is_dir():
            st.error(f"❌ Папка '{image_folder_path}' не существует")
        else:
            st.session_state.is_running = True

            # Create progress container
            progress_container = st.container()
            with progress_container:
                st.markdown("### 🔄 Выполнение оценки")

                progress_bar = st.progress(0)
                status_text = st.empty()
                # status_image = st.empty()
                time_estimate = st.empty()

                try:
                    # Setup
                    df = st.session_state.df
                    df_sample = df.head(num_to_sample)
                    results_list = []

                    vision_llm = get_llm(openrouter_key, model_name)
                    judge_llm = get_llm(openrouter_key, judge_name).with_structured_output(EvaluationResult)

                    total_rows = len(df_sample)
                    start_time = time.time()

                    # Process each row
                    for i, row in enumerate(df_sample.iterrows()):
                        data = row[1]
                        image_name = data['image_name']
                        question = data['question']
                        true_answer = data['answer']
                        image_path = Path(image_folder_path) / image_name

                        if not image_path.exists():
                            st.warning(f"⚠️ Изображение не найдено: {image_path}")
                            continue

                        # Update status
                        status_text.info(f"🔍 Обработка {i + 1}/{total_rows}: {image_name}")
                        # status_image.image(str(image_path), width=500)

                        # Calculate time estimate
                        if i > 0:
                            elapsed = time.time() - start_time
                            avg_time_per_item = elapsed / (i + 1)
                            remaining_items = total_rows - (i + 1)
                            estimated_remaining = avg_time_per_item * remaining_items
                            time_estimate.caption(f"⏱️ Осталось примерно: {estimated_remaining:.0f} сек")

                        # Process image
                        encoded_image = encode_image(image_path)
                        mime_type = f"image/{image_path.suffix[1:]}"

                        vision_message = HumanMessage(
                            content=[
                                {"type": "text", "text": question},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                            ]
                        )

                        item_start = time.time()
                        response = vision_llm.invoke([vision_message])
                        generated_answer = response.content

                        # Judge the answer
                        judge_prompt = judge_prompt_template.format(
                            question=question,
                            true_answer=true_answer,
                            generated_answer=generated_answer
                        )
                        evaluation = judge_llm.invoke(judge_prompt)
                        item_time = time.time() - item_start

                        # Store result
                        result_item = {
                            "image_name": image_name,
                            "question": question,
                            "true_answer": true_answer,
                            "generated_answer": generated_answer,
                            "score": evaluation.score,
                            "justification": evaluation.justification,
                            "time_elapsed": item_time
                        }
                        results_list.append(result_item)

                        # Update progress
                        progress_bar.progress((i + 1) / total_rows)

                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_model_name = model_name.replace("/", "_")
                    benchmark_filename = f"benchmarks/{safe_model_name}_{timestamp}.json"
                    os.makedirs("benchmarks", exist_ok=True)

                    is_sample = num_to_sample < len(df)
                    avg_score = sum(r['score'] for r in results_list) / len(results_list) if results_list else 0

                    final_output = {
                        "meta": {
                            "model_name": model_name,
                            "judge_name": judge_name,
                            "provider_tag": provider_tag,
                            "timestamp": timestamp,
                            "total_items": len(results_list),
                            "average_score": avg_score,
                            "is_sample": is_sample,
                            "sample_size": num_to_sample,
                            "average_time": sum(r['time_elapsed'] for r in results_list) / len(
                                results_list) if results_list else 0
                        },
                        "results": results_list
                    }

                    with open(benchmark_filename, "w", encoding='utf-8') as f:
                        json.dump(final_output, f, indent=4, ensure_ascii=False)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    # status_image.empty()
                    time_estimate.empty()

                    # Store results in session state
                    st.session_state.results = final_output
                    st.session_state.selected_result = final_output

                    # Show success message
                    st.success(f"✅ Оценка завершена! Результаты сохранены в `{benchmark_filename}`")
                    st.balloons()

                    # Display results summary
                    st.markdown("---")
                    st.markdown("## 📊 Сводка результатов")

                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    with col_res1:
                        st.metric("🎯 Средний балл", f"{avg_score:.2f}")
                    with col_res2:
                        st.metric("✅ Оценено", len(results_list))
                    with col_res3:
                        avg_time = sum(r['time_elapsed'] for r in results_list) / len(results_list)
                        st.metric("⏱️ Среднее время", f"{avg_time:.2f}с")
                    with col_res4:
                        perfect_scores = sum(1 for r in results_list if r['score'] == 5)
                        st.metric("⭐ Идеальных", perfect_scores)

                except Exception as e:
                    st.error(f"❌ Произошла ошибка: {e}")
                finally:
                    st.session_state.is_running = False

with tab2:
    st.markdown("### Анализ результатов")

    # Load all available benchmarks
    all_benchmarks = load_all_benchmarks()

    if all_benchmarks:
        # Create options for selectbox
        options = []
        for bench in all_benchmarks:
            meta = bench['meta']
            timestamp = datetime.strptime(meta['timestamp'], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
            sample_info = " (Sample)" if meta.get('is_sample', False) else ""
            option_text = f"{meta['model_name']} - {timestamp} - Балл: {meta['average_score']:.2f}{sample_info}"
            options.append(option_text)

        # Select result to display
        default_index = 0
        if st.session_state.selected_result:
            # Try to find the current result in the list
            for i, bench in enumerate(all_benchmarks):
                if bench['meta']['timestamp'] == st.session_state.selected_result.get('meta', {}).get('timestamp'):
                    default_index = i
                    break

        selected_index = st.selectbox(
            "Выберите результат для просмотра:",
            range(len(options)),
            format_func=lambda x: options[x],
            index=default_index,
            help="По умолчанию показан последний результат"
        )

        # Get selected benchmark
        selected_benchmark = all_benchmarks[selected_index]
        st.session_state.selected_result = selected_benchmark

        # Display selected result
        meta = selected_benchmark['meta']
        results_df = pd.DataFrame(selected_benchmark['results'])

        # Model info card
        with st.expander("ℹ️ Информация о тесте", expanded=True):
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f"**Модель:** {meta['model_name']}")
                st.markdown(f"**Судья:** {meta['judge_name']}")
            with col_info2:
                st.markdown(f"**Провайдер:** {meta.get('provider_tag', 'N/A')}")
                timestamp_str = datetime.strptime(meta['timestamp'], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"**Время теста:** {timestamp_str}")
            with col_info3:
                sample_badge = "🎲 Выборка" if meta.get('is_sample', False) else "✅ Полный"
                st.markdown(f"**Тип:** {sample_badge}")
                st.markdown(f"**Обработано:** {meta['total_items']} вопросов")

        st.markdown("---")

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_score = meta['average_score']
            st.metric("🎯 Средний балл", f"{avg_score:.2f}",
                      delta=f"{avg_score - 3:.2f}" if avg_score != 3 else "0")
        with col2:
            total_evaluated = meta['total_items']
            st.metric("📝 Всего оценено", total_evaluated)
        with col3:
            high_scores = len(results_df[results_df['score'] >= 4])
            st.metric("✨ Высокие оценки", high_scores,
                      delta=f"{(high_scores / total_evaluated * 100):.0f}%")
        with col4:
            avg_time = meta.get('average_time',
                                results_df['time_elapsed'].mean() if 'time_elapsed' in results_df else 0)
            st.metric("⏱️ Среднее время", f"{avg_time:.2f}с")

        st.markdown("---")

        # Visualizations
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            # Score distribution
            fig = create_score_distribution_chart(selected_benchmark['results'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        with col_viz2:
            # Performance gauge
            fig = create_performance_gauge(avg_score)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Detailed Results Table
        st.markdown("### 📋 Детальные результаты")

        # Add filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            score_filter = st.multiselect(
                "Фильтр по оценкам",
                options=[1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5]
            )

        with col_filter2:
            # Search in questions
            search_query = st.text_input("🔍 Поиск в вопросах", "")

        with col_filter3:
            # Sort options
            sort_by = st.selectbox(
                "Сортировка",
                options=["score", "time_elapsed", "image_name"],
                format_func=lambda x: {
                    "score": "По оценке",
                    "time_elapsed": "По времени",
                    "image_name": "По изображению"
                }[x]
            )

        # Apply filters
        filtered_df = results_df[results_df['score'].isin(score_filter)]
        if search_query:
            filtered_df = filtered_df[filtered_df['question'].str.contains(search_query, case=False, na=False)]

        # Sort
        ascending = sort_by == "image_name"
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        # Display dataframe
        st.dataframe(
            filtered_df[['image_name', 'question', 'score', 'justification', 'time_elapsed']],
            use_container_width=True,
            height=400
        )

        # Export options
        st.markdown("### 💾 Экспорт результатов")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"results_{meta['model_name'].replace('/', '_')}_{meta['timestamp']}.csv",
                mime="text/csv"
            )

        with col_exp2:
            json_str = json.dumps(selected_benchmark, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 Скачать JSON",
                data=json_str,
                file_name=f"results_{meta['model_name'].replace('/', '_')}_{meta['timestamp']}.json",
                mime="application/json"
            )
    else:
        st.info("📊 Нет доступных результатов. Запустите оценку для просмотра результатов.")

with tab3:
    # Load all benchmarks
    all_benchmarks = load_all_benchmarks()

    if all_benchmarks and len(all_benchmarks) > 1:
        # Create summary dataframe
        summary_data = []
        for bench in all_benchmarks:
            meta = bench['meta']
            results = bench['results']

            summary_data.append({
                'model': meta['model_name'],
                'judge': meta['judge_name'],
                'provider': meta.get('provider_tag', 'Unknown'),
                'timestamp': datetime.strptime(meta['timestamp'], "%Y%m%d_%H%M%S"),
                'avg_score': meta['average_score'],
                'total_items': meta['total_items'],
                'is_sample': meta.get('is_sample', False),
                'avg_time': meta.get('average_time',
                                     sum(r['time_elapsed'] for r in results) / len(results) if results else 0)
            })

        summary_df = pd.DataFrame(summary_data)

        # Comparison mode selector
        st.markdown("#### 🎯 Режим сравнения")
        comparison_mode = st.radio(
            "Выберите режим:",
            ["Все модели", "Выбранные модели"],
            horizontal=True
        )

        if comparison_mode == "Выбранные модели":
            # Model selection
            unique_models = summary_df['model'].unique()
            selected_models = st.multiselect(
                "Выберите модели для сравнения:",
                options=unique_models,
                default=list(unique_models[:3]) if len(unique_models) >= 3 else list(unique_models)
            )

            # Filter dataframe
            comparison_df = summary_df[summary_df['model'].isin(selected_models)]
        else:
            comparison_df = summary_df
            selected_models = summary_df['model'].unique()

        if len(selected_models) > 0:
            st.markdown("---")

            # Summary statistics
            st.markdown("#### 📊 Сводная статистика")

            # Group by model and calculate aggregates
            model_stats = comparison_df.groupby('model').agg({
                'avg_score': ['mean', 'min', 'max', 'count'],
                'avg_time': 'mean',
                'total_items': 'sum'
            }).round(2)

            model_stats.columns = ['Средний балл', 'Мин. балл', 'Макс. балл', 'Кол-во тестов', 'Среднее время',
                                   'Всего вопросов']

            # Display as styled dataframe
            st.dataframe(
                model_stats.style.background_gradient(subset=['Средний балл'], cmap='RdYlGn', vmin=1, vmax=5),
                use_container_width=True
            )

            st.markdown("---")

            # Visualization
            st.markdown("#### 📈 Визуализация")

            # Prepare data for comparison chart
            chart_data = []
            for model in selected_models:
                model_data = comparison_df[comparison_df['model'] == model]
                chart_data.append({
                    'model': model,
                    'avg_score': model_data['avg_score'].mean(),
                    'avg_time': model_data['avg_time'].mean()
                })

            # Create comparison chart
            fig = create_comparison_chart(chart_data, include_time=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Detailed comparison table
            st.markdown("#### 📋 Детальное сравнение")

            # Add filters
            col1, col2 = st.columns(2)
            with col1:
                show_samples = st.checkbox("Показать тесты-выборки", value=True)
            with col2:
                pass

            # Apply filters
            if not show_samples:
                comparison_df = comparison_df[~comparison_df['is_sample']]


            # Format for display
            display_df = comparison_df.copy()
            display_df['Дата'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['Тип'] = display_df['is_sample'].apply(lambda x: '🎲 Выборка' if x else '✅ Полный')
            display_df['Средний балл'] = display_df['avg_score'].round(2)
            display_df['Время (сек)'] = display_df['avg_time'].round(2)

            # Display table
            st.dataframe(
                display_df[['Дата', 'model', 'provider', 'Тип', 'Средний балл', 'Время (сек)', 'total_items']].rename(
                    columns={
                        'model': 'Модель',
                        'provider': 'Провайдер',
                        'total_items': 'Вопросов'
                    }),
                use_container_width=True,
                height=400
            )

            # Export comparison
            st.markdown("#### 💾 Экспорт сравнения")

            export_data = display_df[['Дата', 'model', 'provider', 'Тип', 'Средний балл', 'Время (сек)', 'total_items']]
            csv = export_data.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Скачать сравнение (CSV)",
                data=csv,
                file_name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        else:
            st.warning("⚠️ Выберите хотя бы одну модель для сравнения")

    elif len(all_benchmarks) == 1:
        st.info("📊 Для сравнения необходимо минимум 2 результата тестирования")
    else:
        st.info("📊 Нет доступных результатов для сравнения. Запустите тестирование моделей.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #718096; padding: 2rem 0;'>
        <p>Vision Eval</p>
    </div>
    """,
    unsafe_allow_html=True
)
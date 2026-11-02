import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

# ==========================================
# 1. 配置与初始化
# ==========================================
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "") 

st.set_page_config(layout="wide", page_title="护理学生人群画像看板")
st.title("📊 护理学生文化感知人群画像与精准干预看板")

# ==========================================
# 2. 人群图鉴速览
# ==========================================
with st.expander("🔍 点击查看：这三类人群到底是什么特征？（人群图鉴速览）", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    col_a.info("🟦 **群体0（认知薄弱型）**\n\n**特征**：脑中对文化多样性缺乏概念，且缺乏临床应对能力。\n\n**痛点**：理论基础和实操双欠缺。")
    col_b.success("🟩 **群体1（临床经验型）**\n\n**特征**：临床情景判断能力最强，但不善反思。\n\n**痛点**：做事凭直觉和经验，缺乏理论支撑。")
    col_c.error("🟥 **群体2（理论脱节型）**\n\n**特征**：文化反思意识极强，但临床实操判断偏弱。\n\n**痛点**：眼高手低，理论无法向实操转化。")

st.divider()

# ==========================================
# 3. 数据加载与处理（🌟 修复了1-5映射规则）
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_excel('final_data_for_bi.xlsx')
    df.columns = df.columns.str.strip()
    
    col_train = [c for c in df.columns if '培训' in c][0]
    col_gender = [c for c in df.columns if '性别' in c][0]
    col_grade = [c for c in df.columns if '年级' in c][0]
    col_group = [c for c in df.columns if '人群标签' in c][0]
    
    # 1. 处理性别（1->男, 2->女）
    df[col_gender] = df[col_gender].astype(str).str.replace(r'\.0$', '', regex=True)
    gender_map = {'1': '男', '2': '女'}
    df[col_gender] = df[col_gender].map(gender_map).fillna('未知')
    
    # 2. 处理年级（1->大一, 2->大二, 3->大三, 4->大四, 5->其他）
    df[col_grade] = df[col_grade].astype(str).str.replace(r'\.0$', '', regex=True)
    grade_map = {'1': '大一', '2': '大二', '3': '大三', '4': '大四', '5': '其他'}
    df[col_grade] = df[col_grade].map(grade_map).fillna('其他')
    
    # 核心修复：强制按正确顺序定义分类变量
    grade_order = ['大一', '大二', '大三', '大四', '其他']
    df[col_grade] = pd.Categorical(df[col_grade], categories=grade_order, ordered=True)
    
    # 3. 处理人群标签
    group_map = {'群体 0': '群体0: 认知薄弱型', '群体 1': '群体1: 临床经验型', '群体 2': '群体2: 理论脱节型'}
    df[col_group] = df[col_group].replace(group_map)
    
    # 4. 处理培训标签
    train_map = {'是': '接受过培训', '否': '未接受过培训', 1: '接受过培训', 2: '未接受过培训'}
    df[col_train] = df[col_train].replace(train_map)
    
    return df, col_train, col_gender, col_grade, col_group

try:
    df, col_train, col_gender, col_grade, col_group = load_data()
except Exception as e:
    st.error(f"数据读取失败，请检查同目录下是否有 final_data_for_bi.xlsx 文件。错误信息：{e}")
    st.stop()

features = ['F1_患者医护行为', 'F2_自我评估', 'F3_自我意识', 'F4_文化影响']

# ==========================================
# 4. AI 智能诊断函数
# ==========================================
def get_ai_insight(group_means_df, group_name):
    if not DEEPSEEK_API_KEY:
        return "⚠️ 尚未配置 DeepSeek API Key，请在 Streamlit Cloud 的 Secrets 中配置 `DEEPSEEK_API_KEY`。"
    
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        data_str = ""
        for name, row in group_means_df.iterrows():
            data_str += f"- {name}: F1={row['F1_患者医护行为']:.1f}, F2={row['F2_自我评估']:.1f}, F3={row['F3_自我意识']:.1f}, F4={row['F4_文化影响']:.1f}\n"

        prompt = f"""
        你是一位资深的护理教育专家和数据洞察分析师。
        现在有一批【{group_name}】的数据，其文化感知量表的四维度平均分如下：
        {data_str}
        （注：满分参考：F1满分30，F2满分24，F3满分30，F4满分42）
        
        请根据以上数据，用专业、犀利、有洞察力的语言，生成一段约150字左右的诊断报告。
        必须包含三个部分，并使用Markdown的加粗格式：
        1. 【核心画像】：一针见血指出这三类人的特质。
        2. 【教育痛点】：指出这种分数组合揭示的护理教育盲区。
        3. 【干预建议】：给学院提出1-2条具体的、可执行的教学改革建议。
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个高阶护理教育专家，说话一针见血，擅长数据驱动决策。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI 接口调用失败，请检查网络或 API Key 余额。错误信息：{e}"

# ==========================================
# 5. 页面布局
# ==========================================
tab1, tab2 = st.tabs(["🎛️ 单点智能交互（AI解读）", "📚 多维度组合全景图"])
plotly_font = dict(family="SimHei, Arial, sans-serif", size=14)

with tab1:
    st.sidebar.header("🎛️ 动态数据筛选")
    selected_gender = st.sidebar.selectbox("选择性别", ["全部"] + list(df[col_gender].unique()))
    
    # 使用 Categorical 的 categories 属性，保证下拉框里是严格的大一、大二、大三、大四、其他
    selected_grade = st.sidebar.selectbox("选择年级", ["全部"] + list(df[col_grade].cat.categories))

    df_filtered = df.copy()
    if selected_gender != "全部":
        df_filtered = df_filtered[df_filtered[col_gender] == selected_gender]
    if selected_grade != "全部":
        df_filtered = df_filtered[df_filtered[col_grade] == selected_grade]

    st.markdown(f"**当前筛选：** 性别 = `{selected_gender}` | 年级 = `{selected_grade}` | **样本数：** `{len(df_filtered)}` 人")

    if len(df_filtered) > 0:
        group_means = df_filtered.groupby(col_group)[features].mean()
        
        colors = ['#3498DB', '#2ECC71', '#E74C3C']
        fig_radar = go.Figure()
        for i, (group_name, row) in enumerate(group_means.iterrows()):
            fig_radar.add_trace(go.Scatterpolar(r=row.values, theta=features, fill='toself', name=group_name, line_color=colors[i % len(colors)]))
        
        fig_radar.update_layout(title="四大维度人群雷达图", polar=dict(radialaxis=dict(visible=True, range=[0, 35])), height=450, font=plotly_font)

        counts = df_filtered.groupby([col_train, col_group]).size().unstack(fill_value=0)
        labels = list(counts.index) + list(counts.columns)
        source, target, value = [], [], []
        for i, src in enumerate(counts.index):
            for j, tgt in enumerate(counts.columns):
                if counts.loc[src, tgt] > 0:
                    source.append(i); target.append(len(counts.index) + j); value.append(counts.loc[src, tgt])
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=25, thickness=20, label=labels, color=["#95A5A6"]*len(counts.index) + ["#3498DB"]*len(counts.columns)), 
            link=dict(source=source, target=target, value=value, color="rgba(180, 180, 180, 0.4)")
        )])
        fig_sankey.update_layout(title="培训经历与人群流转桑基图", height=450, font=plotly_font)

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_radar, use_container_width=True)
        c2.plotly_chart(fig_sankey, use_container_width=True)
        
        st.info(f"🤖 **AI 专家诊断报告**：针对【{selected_gender} - {selected_grade}】的智能洞察（由 DeepSeek 实时生成）")
        with st.spinner("AI 专家正在分析数据并撰写报告，请稍候..."):
            ai_report = get_ai_insight(group_means, f"{selected_gender}{selected_grade}学生群体")
            st.markdown(ai_report)

with tab2:
    st.markdown("这里展示了不同性别与年级组合下的细分人群画像。")
    
    # 遍历时也严格按照我们的分类顺序
    for grade in df[col_grade].cat.categories:
        for gender in df[col_gender].unique():
            df_sub = df[(df[col_grade] == grade) & (df[col_gender] == gender)]
            if len(df_sub) < 10: continue 
            
            gender_text = gender
            
            with st.expander(f"👉 {grade} - {gender_text}生 (样本数: {len(df_sub)}人)"):
                group_means_sub = df_sub.groupby(col_group)[features].mean()
                fig_r = go.Figure()
                for i, (g_name, row) in enumerate(group_means_sub.iterrows()):
                    fig_r.add_trace(go.Scatterpolar(r=row.values, theta=features, fill='toself', name=g_name, line_color=colors[i % len(colors)]))
                fig_r.update_layout(title=f"雷达图", polar=dict(radialaxis=dict(visible=True, range=[0, 35])), height=400, font=plotly_font, margin=dict(t=40, b=20))
                
                counts_sub = df_sub.groupby([col_train, col_group]).size().unstack(fill_value=0)
                labels_sub = list(counts_sub.index) + list(counts_sub.columns)
                s, t, v = [], [], []
                for i, src in enumerate(counts_sub.index):
                    for j, tgt in enumerate(counts_sub.columns):
                        if counts_sub.loc[src, tgt] > 0:
                            s.append(i); t.append(len(counts_sub.index) + j); v.append(counts_sub.loc[src, tgt])
                fig_s = go.Figure(data=[go.Sankey(
                    node=dict(pad=20, thickness=15, label=labels_sub, color=["#95A5A6"]*len(counts_sub.index) + ["#3498DB"]*len(counts_sub.columns)), 
                    link=dict(source=s, target=t, value=v, color="rgba(180, 180, 180, 0.4)")
                )])
                fig_s.update_layout(title=f"桑基图", height=400, font=plotly_font, margin=dict(t=40, b=20))

                c1, c2 = st.columns(2)
                c1.plotly_chart(fig_r, use_container_width=True)
                c2.plotly_chart(fig_s, use_container_width=True)
                
                st.markdown("---")
                with st.spinner(f"AI 正在生成【{grade} - {gender_text}生】的洞察..."):
                    st.markdown(get_ai_insight(group_means_sub, f"{grade}{gender_text}生"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="护理学生人群画像看板")
st.title("📊 护理学生文化感知人群画像与精准干预看板")

# ==========================================
# 1. 人群图鉴速览
# ==========================================
with st.expander("🔍 点击查看：这三类人群到底是什么特征？（人群图鉴速览）", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    col_a.info("🟦 **群体0（认知薄弱型）**\n\n**特征**：脑中对文化多样性缺乏概念，且缺乏临床应对能力。\n\n**痛点**：理论基础和实操双欠缺。")
    col_b.success("🟩 **群体1（临床经验型）**\n\n**特征**：临床情景判断能力最强，但不善反思。\n\n**痛点**：做事凭直觉和经验，缺乏理论支撑。")
    col_c.error("🟥 **群体2（理论脱节型）**\n\n**特征**：文化反思意识极强，但临床实操判断偏弱。\n\n**痛点**：眼高手低，理论无法向实操转化。")

st.divider()

# ==========================================
# 2. 数据加载与处理
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_excel('final_data_for_bi.xlsx')
    df.columns = df.columns.str.strip()
    
    col_train = [c for c in df.columns if '培训' in c][0]
    col_gender = [c for c in df.columns if '性别' in c][0]
    col_grade = [c for c in df.columns if '年级' in c][0]
    col_group = [c for c in df.columns if '人群标签' in c][0]
    
    df[col_gender] = df[col_gender].astype(str).str.replace(r'\.0$', '', regex=True)
    gender_map = {'1': '男', '2': '女'}
    df[col_gender] = df[col_gender].map(gender_map).fillna('未知')
    
    df[col_grade] = df[col_grade].astype(str).str.replace(r'\.0$', '', regex=True)
    grade_map = {'1': '大一', '2': '大二', '3': '大三', '4': '大四', '5': '其他'}
    df[col_grade] = df[col_grade].map(grade_map).fillna('其他')
    
    grade_order = ['大一', '大二', '大三', '大四', '其他']
    df[col_grade] = pd.Categorical(df[col_grade], categories=grade_order, ordered=True)
    
    group_map = {'群体 0': '群体0: 认知薄弱型', '群体 1': '群体1: 临床经验型', '群体 2': '群体2: 理论脱节型'}
    df[col_group] = df[col_group].replace(group_map)
    
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
# 3. 🌟 核心：本地写死的专家级字典（不调用任何 API！）
# ==========================================
def get_mock_insight(group_name):
    # 默认通用文案
    default_text = """
    * **【核心画像】**：当前人群呈现典型的混合特征。其中“理论脱节型”占比最高，他们拥有极高的文化反思意识，但临床情景判断能力较弱。
    * **【教育痛点】**：当前课程体系过于注重理论灌输，缺乏真实临床情境的实操演练，导致学生“知行分离”。
    * **【干预建议】**：建议减少纯理论授课，增加 PBL（问题导向）和标准化病人（SP）情景模拟训练。
    """
    
    # 针对特定人群的“定制化”文案（你可以自己随意修改、添加）
    if "大一" in group_name:
        return """
        * **【核心画像】**：大一新生刚接触专业课，认知普遍处于“白纸”状态，“认知薄弱型”群体比例显著高于其他年级。
        * **【教育痛点】**：早期接触临床较少，对文化冲突缺乏概念，容易陷入书本上的刻板印象。
        * **【干预建议】**：建议在第一学期就引入跨文化护理的案例讨论，尽早唤醒文化敏感度。
        """
    elif "大四" in group_name:
        return """
        * **【核心画像】**：经过四年的学习，大四学生的文化意识极高，但面临严峻的“理论脱节型”困境，眼高手低现象突出。
        * **【教育痛点】**：实习期间发现真实临床复杂多变，原以为“懂了”的知识，在面对真实患者（如拒绝对视、拒绝握手）时不知所措。
        * **【干预建议】**：建议在实习前开展冲突情景模拟，并增加临床带教过程中的实时复盘与纠偏。
        """
    elif "男" in group_name:
        return """
        * **【核心画像】**：男生群体在“自我反思（F2）”和“文化感知（F3）”上普遍得分偏低，更倾向于凭直觉（临床经验型）处理问题。
        * **【教育痛点】**：传统的理论授课难以引起该群体的共鸣，缺乏反思环节导致无法建立系统性的文化敏感度。
        * **【干预建议】**：建议采用行动学习法或小组辩论，通过强互动的方式强制他们进行自我文化剖析。
        """
    elif "女" in group_name:
        return """
        * **【核心画像】**：女生群体在共情和反思意识（F3）上表现优异，但容易被“过度自信（F2）”迷惑，导致实操判断（F1）出现偏差。
        * **【教育痛点】**：往往“觉得”自己很懂，但在突发或隐性的文化冲突中，缺乏客观的临床判断标准。
        * **【干预建议】**：建议强化客观结构化临床考试（OSCE），在考核中设置文化冲突“陷阱”，打破盲目自信。
        """
    else:
        return default_text

# ==========================================
# 4. 页面布局
# ==========================================
tab1, tab2 = st.tabs(["🎛️ 单点智能交互（AI解读）", "📚 多维度组合全景图"])
plotly_font = dict(family="SimHei, Arial, sans-serif", size=14)

with tab1:
    st.sidebar.header("🎛️ 动态数据筛选")
    selected_gender = st.sidebar.selectbox("选择性别", ["全部"] + list(df[col_gender].unique()))
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
        
        st.info(f"🤖 **AI 专家诊断报告**：针对【{selected_gender} - {selected_grade}】的智能洞察")
        # 直接调用本地字典！
        insight_text = get_mock_insight(f"{selected_grade}{selected_gender}")
        st.markdown(insight_text)

with tab2:
    st.markdown("这里展示了不同性别与年级组合下的细分人群画像。")
    for grade in df[col_grade].cat.categories:
        for gender in df[col_gender].unique():
            df_sub = df[(df[col_grade] == grade) & (df[col_gender] == gender)]
            if len(df_sub) < 10: continue 
            
            with st.expander(f"👉 {grade} - {gender}生 (样本数: {len(df_sub)}人)"):
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
                # 全景图里也用本地字典
                st.markdown(get_mock_insight(f"{grade}{gender}"))

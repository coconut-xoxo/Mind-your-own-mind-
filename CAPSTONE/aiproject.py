# Student Wellness Dashboard — Streamlit Polished (Bright & Colourful)
# Save this file as app.py and run with streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import sqlite3
from datetime import datetime
from streamlit_option_menu import option_menu
DATA_FILE = "wellness_data.csv"
df = pd.read_csv(DATA_FILE) # This creates the 'df' variable
# ===============================
# Function to generate detailed AI insights
# ===============================
def generate_detailed_insight(row, df=None):
    """
    Generates a detailed, paragraph-style AI insight for a single row.
    Each insight includes:
    - Observation (what)
    - Importance (why)
    - Actionable advice (how)
    - Optional trend / comparison context
    """
    date = row.get('Date', 'Unknown')
    mood = row.get('Mood', 0)
    sleep = row.get('Sleep', 0)
    stress = row.get('Stress', 0)
    sentiment = row.get('Sentiment', 0)
    screentime = row.get('ScreenTime', 0)

    insight = f"📅 Date: {date}\n\n"

    # Paragraph 1: Overview
    insight += (
        f"On this day, your recorded mood was {mood}/5, you slept for {sleep} hours, "
        f"experienced a stress level of {stress}/10, spent approximately {screentime} hours on screens, "
        f"and your written notes suggest a sentiment score of {sentiment:.2f}. "
        f"This gives us a snapshot of your emotional and physical state for the day.\n\n"
    )

    # Paragraph 2: Mood Analysis
    if mood < 3:
        insight += (
            f"Your mood is noticeably low. Sustained low mood can impact concentration, motivation, and overall well-being. "
            f"Understanding the reasons behind this dip is important. Consider reflecting on what might have triggered these feelings today. "
            f"Reaching out to a trusted friend, family member, or counselor can provide support and perspective.\n\n"
        )
    elif mood == 5:
        insight += (
            f"Your mood is excellent today! Maintaining this positive emotional state is important for productivity and resilience. "
            f"Continue engaging in activities that make you feel fulfilled and energized.\n\n"
        )
    else:
        insight += (
            f"Your mood today is moderate. Keep observing patterns to see if certain activities or habits consistently affect how you feel. "
            f"Awareness is the first step toward emotional balance.\n\n"
        )

    # Paragraph 3: Sleep Analysis
    if sleep < 6:
        insight += (
            f"Sleep duration is below the recommended threshold. Insufficient sleep can reduce alertness, impair memory, "
            f"and amplify stress or negative emotions. Prioritize establishing a consistent sleep schedule, limit screen exposure before bedtime, "
            f"and consider relaxing routines such as reading or light stretching to improve sleep quality.\n\n"
        )
    elif sleep > 9:
        insight += (
            f"Sleep duration is quite high today. While rest is important, unusually long sleep may indicate fatigue or underlying stress. "
            f"Monitor your energy levels and ensure that sleep quality is restorative.\n\n"
        )
    else:
        insight += (
            f"Sleep duration is within a healthy range. Maintaining consistent sleep helps stabilize mood and cognitive performance.\n\n"
        )

    # Paragraph 4: Stress & Screen Time
    if stress > 6:
        insight += (
            f"Stress levels are elevated today. Chronic high stress can affect both physical and mental health. "
            f"Consider brief relaxation exercises, meditation, or short walks to reduce tension. "
            f"Balancing workload and taking intentional breaks can also help mitigate stress accumulation.\n\n"
        )
    if screentime > 5:
        insight += (
            f"Screen time is high, which can contribute to eye strain, fatigue, and heightened stress. "
            f"Try breaking tasks into screen-free intervals, engaging in offline hobbies, or limiting social media usage. "
            f"These small changes can improve focus and mental clarity.\n\n"
        )

    # Paragraph 5: Sentiment vs Mood
    if (sentiment < -0.2 and mood >= 4) or (sentiment > 0.2 and mood <= 2):
        insight += (
            f"There appears to be a mismatch between your textual sentiment and self-reported mood. "
            f"This could indicate underlying emotions that aren’t immediately visible in your self-report. "
            f"Reflecting on this discrepancy can help you better understand and articulate your feelings, "
            f"improving self-awareness over time.\n\n"
        )

    # Paragraph 6: Trend / optional comparison
    if df is not None and len(df) >= 7:
        recent_mood = df['Mood'].tail(7).mean()
        if mood < recent_mood - 1:
            insight += (
                f"Compared to the past week, your mood today is lower than your recent average ({recent_mood:.1f}/5). "
                f"Noticing these trends helps identify patterns and potential triggers, allowing for more proactive mental health care.\n\n"
            )

    insight += "✅ Keep logging consistently. Regular entries help detect meaningful patterns and allow AI to provide more personalized and actionable insights over time.\n"

    return insight


conn = sqlite3.connect("wellness.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS wellness (
    date TEXT,
    mood INTEGER,
    sleep REAL,
    stress INTEGER,
    screen REAL,
    sentiment REAL
)
""")
conn.commit()



# -----------------------------
# Config & Constants
# -----------------------------
st.set_page_config(page_title="Mind Your Own Mind", layout="wide", initial_sidebar_state="expanded")


# -----------------------------
# Utilities: load/save CSV and sentiment
# -----------------------------

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors= 'coerce', format='mixed')
        return df
    return pd.DataFrame(columns=["Date", "Mood", "Sleep", "ScreenTime", "Exercise", "Stress", "Journal", "Sentiment"])


def save_data(df):
    df.to_csv(DATA_FILE, index=False)


def sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return TextBlob(text).sentiment.polarity

# -----------------------------
# Small CSS to make it bright & colourful
# -----------------------------
st.markdown(
    "<style>\n    .stApp { background: linear-gradient(135deg, #FFF7ED 0%, #FEE2E2 50%, #ECFEFF 100%);}\n    .card { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);}\n    .small { font-size: 0.9rem; color: #444; }\n    </style>",
    unsafe_allow_html=True,
)

# -----------------------------
# App state
# -----------------------------
data = load_data()
data = load_data()
df = pd.DataFrame(data)


# -----------------------------
# Top navigation using option_menu
# -----------------------------
# Remove top padding above sidebar logo
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        padding-top: 0rem !important;
    }
    [data-testid="stSidebar"] img {
        margin-top: -30px !important;  /* adjust: try -20 or -30 if needed */
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    # Logo section
    st.image("CAPSTONE/logo.jpg", use_container_width=True)
    st.markdown("<h2 style='text-align: center; margin-bottom: -10px;'>MindSync</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

    # Navigation menu
    choice = option_menu(
        None,
        ["Home", "Daily Log", "Visual Insights", "AI Insights", "Premium"],
        icons=["house", "journal-text", "bar-chart-line", "robot", "star"],
        menu_icon="cast",
        default_index=0,
    )

    # Footer
    st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:11px; color:gray;'>Made for students • Bright & Colourful demo</p>", unsafe_allow_html=True)

# -----------------------------
# Home
# -----------------------------
if choice == "Home":
    st.header("🌈 Mind Your Own Mind")
    st.markdown("Welcome — log daily wellness, explore interactive charts, and get simple AI insights. This demo stores data locally in a CSV called wellness_data.csv.")

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.metric("Entries", len(data), delta=None)
    with col2:
        if not data.empty:
            avg_mood = data['Mood'].mean()
            avg_sleep = data['Sleep'].mean()
            st.metric("Avg Mood", f"{avg_mood:.2f} / 5", delta=None)
            st.metric("Avg Sleep", f"{avg_sleep:.1f} hrs", delta=None)
        else:
            st.info("No data yet — go to 'Daily Log' to add your first entry")
    with col3:
        st.metric("Premium Ideas", "Available", delta=None)

    st.markdown("---")
    st.subheader("How this helps")
    st.markdown("- Quick daily logging that takes 30 seconds.\n- Interactive trends to spot patterns (e.g., low sleep → low mood).\n- Simple, interpretable AI features to showcase data science in psychology.")

# -----------------------------
# Daily Log
# -----------------------------
elif choice == "Daily Log":
    st.header("📝 Daily Log")
    st.markdown("Enter today's wellness details. The journal entry is optional but useful for sentiment analysis.")

    with st.form(key='entry_form'):

        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input('Date', value=datetime.today())
            mood = st.slider('Mood (1–5)', 1, 5, 3)
            sleep = st.number_input('Sleep (hours)', min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        with col2:
            screen = st.number_input('Screen time (hours)', min_value=0.0, max_value=24.0, value=3.0, step=0.25)
            exercise = st.number_input('Exercise (minutes)', min_value=0, max_value=300, value=30)
            stress = st.slider('Stress (1–10)', 1, 10, 5)
        with col3:
            journal = st.text_area('Journal (optional)', placeholder='How was your day?')
            st.markdown('<div class="small">Tip: Short phrase notes like "exam stress" or "family time" are useful.</div>', unsafe_allow_html=True)

        submit = st.form_submit_button('Save Entry')

    if submit:
        sentiment = sentiment_score(journal)
        new = pd.DataFrame([[date, mood, sleep, screen, exercise, stress, journal, sentiment]],
                           columns=['Date','Mood','Sleep','ScreenTime','Exercise','Stress','Journal','Sentiment'])
        data = pd.concat([data, new], ignore_index=True)
        save_data(data)
        st.success('Saved! Your entry was added.')

    st.markdown('### Recent entries')
    if data.empty:
        st.info('No entries yet.')
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        st.dataframe(data.sort_values('Date', ascending=False).reset_index(drop=True))


# -----------------------------
# Visual Insights (Plotly)
# -----------------------------
elif choice == "Visual Insights":
    st.header("📈 Visual Insights")
    if data.empty:
        st.info('Add entries first in the Daily Log to see visual insights.')
    else:
        df = data.sort_values('Date')
        # Mood over time (interactive)
        st.subheader('Mood over Time')
        fig = px.line(df, x='Date', y='Mood', markers=True, title='Mood Over Time', range_y=[1,5])
        fig.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Sleep vs Mood scatter
        st.subheader('Sleep vs Mood')
        fig2 = px.scatter(df, x='Sleep', y='Mood', size='Exercise', color='Stress', hover_data=['Date','Journal'], title='Sleep vs Mood')
        st.plotly_chart(fig2, use_container_width=True)

        # Screen time & Stress bar (last 14)
        st.subheader('Screen Time & Stress (Recent)')
        recent = df.tail(14)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=recent['Date'].dt.strftime('%b %d'), y=recent['ScreenTime'], name='Screen (hrs)'))
        fig3.add_trace(go.Bar(x=recent['Date'].dt.strftime('%b %d'), y=recent['Stress'], name='Stress (1-10)'))
        fig3.update_layout(barmode='group', title='Recent Screen Time vs Stress')
        st.plotly_chart(fig3, use_container_width=True)

        # Correlation heatmap
        st.subheader('Correlation Matrix')
        corr = df[['Mood','Sleep','ScreenTime','Exercise','Stress','Sentiment']].corr()
        fig4 = px.imshow(corr, text_auto=True, title='Correlation between measures')
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# AI Insights
# -----------------------------
elif choice == "AI Insights":
    st.header("🤖 AI Insights & Suggestions")

    if data.empty:
        st.info("Add more entries to unlock AI insights.")
    else:
        # Sort data
        df = data.sort_values("Date").copy()
        
        # Generate detailed AI insights
        df["Detailed_AI_Insight"] = df.apply(lambda row: generate_detailed_insight(row, df), axis=1)
        
        # Display latest AI insight
        st.subheader("Latest AI Insight")
        st.write(df["Detailed_AI_Insight"].iloc[-1])

        # Optional: Pattern-based suggestions (like low mood streak, sleep correlation)
        insights = []

        # Rule: 3 consecutive low mood days
        if len(df) >= 3:
            last3 = df['Mood'].tail(3)
            if all(last3 < 3):
                insights.append((
                    'Low mood streak',
                    "You've reported low mood for several days. Consider reaching out to a friend, family member, or counselor."
                ))

        # Sleep correlation
        if df['Sleep'].mean() < 6.5 and df['Mood'].mean() < 3.5:
            insights.append((
                'Sleep & Mood',
                'Lower sleep appears alongside lower mood. Try prioritizing consistent sleep hours.'
            ))

        # Screen time
        if df['ScreenTime'].mean() > 5 and df['Stress'].mean() > 6:
            insights.append((
                'Screen & Stress',
                'High average screen time is associated with higher stress. Introduce short screen breaks.'
            ))

        # Display pattern-based insights
        st.subheader("Detected Patterns / Suggestions")
        if not insights:
            st.success("No concerning patterns detected — keep logging consistently!")
        else:
            for title, text in insights:
                st.warning(f"**{title}**: {text}")

        # Optional: Simple Mood Prediction Demo
        if len(df) >= 5:
            st.markdown('### Mood Prediction (demo)')
            X = df[['Sleep']].values
            y = df['Mood'].values
            model = LinearRegression().fit(X, y)
            next_sleep = st.number_input(
    'If you sleep (hrs) tomorrow...', 
    min_value=0.0, 
    max_value=24.0, 
    value=7.0, 
    step=0.5,
    key="next_sleep_input" )

            pred = model.predict([[next_sleep]])[0]
            st.info(f'Predicted mood (1–5): {pred:.2f} based on sleep hours using a simple linear model')

        # =========================
        # Pattern-based insights
        # =========================

        # Rule: 3 consecutive low mood days
        if len(df) >= 3:
            last3 = df['Mood'].tail(3)
            if all(last3 < 3):
                insights.append((
                    'Low mood streak',
                    "You've reported low mood for several days. Consider reaching out to a friend, family member, or counselor."
                ))

        # Sleep correlation
        if df['Sleep'].mean() < 6.5 and df['Mood'].mean() < 3.5:
            insights.append((
                'Sleep & Mood',
                'Lower sleep appears alongside lower mood. Try prioritizing consistent sleep hours.'
            ))

        # Screen time
        if df['ScreenTime'].mean() > 5 and df['Stress'].mean() > 6:
            insights.append((
                'Screen & Stress',
                'High average screen time is associated with higher stress. Introduce short screen breaks.'
            ))

        # Sentiment mismatch
        recent = df.tail(7)
        mismatch = []
        for _, row in recent.iterrows():
            if (row['Sentiment'] < -0.2 and row['Mood'] >= 4) or (row['Sentiment'] > 0.2 and row['Mood'] <= 2):
                mismatch.append((row['Date'], row['Mood'], row['Sentiment']))
        if mismatch:
            insights.append((
                'Sentiment vs Mood mismatch',
                f'{len(mismatch)} recent entries show mismatch between text sentiment and reported mood.'
            ))

        # Display all collected insights
        if not insights:
            st.success('No concerning patterns detected — keep logging consistently!')
        else:
            for title, text in insights:
                st.warning(f"**{title}**: {text}")

        if len(df) >= 5:
            st.markdown('### Mood Prediction (demo)')
            X = df[['Sleep']].values
            y = df['Mood'].values
            model = LinearRegression().fit(X, y)
            next_sleep = st.number_input(
                'If you sleep (hrs) tomorrow...', min_value=0.0, max_value=24.0, value=7.0, step=0.5
            )
            pred = model.predict([[next_sleep]])[0]
            st.info(f'Predicted mood (1–5): {pred:.2f} based on sleep hours using a simple linear model')

# -----------------------------
# Premium / Business ideas
# -----------------------------
elif choice == "Premium":
    st.header("✨ Premium Features (Mock)")
    st.markdown("This section shows ideas you could build for a paid tier or institutional product:")
    st.markdown("- Weekly PDF summaries with embedded charts and recommendations\n- Personalized AI coach (chat + action plans)\n- School-wide dashboards (anonymized) for early intervention\n- Export to Google Sheets / Google Drive sync")

    st.markdown('### Mock: Weekly Summary')
    if data.empty:
        st.info('Add data to create summaries.')
    else:
        df = data.sort_values('Date')
        last7 = df.tail(7)
        summary = {
            'avg_mood': last7['Mood'].mean(),
            'avg_sleep': last7['Sleep'].mean(),
            'avg_stress': last7['Stress'].mean(),
            'total_exercise_mins': last7['Exercise'].sum()
        }
        st.json(summary)
        st.markdown('Note: Implementing real premium features would require user auth and backend storage.')

# -----------------------------
# Footer: data management controls
# -----------------------------
st.markdown('---')
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    if st.button('Download CSV'):
        if not data.empty:
            st.download_button('Download data', data.to_csv(index=False), file_name='wellness_data.csv')
        else:
            st.info('No data to download.')
with col_b:
    if st.button("Clear All Data"):
        cursor.execute("DELETE FROM wellness")
        conn.commit()
        st.success("All stored data has been permanently cleared.")

with col_c:
    st.markdown('Built for capstone — customize visuals, sentiment model, and backend for production.')




































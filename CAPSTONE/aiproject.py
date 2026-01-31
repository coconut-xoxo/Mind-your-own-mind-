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
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=[
        "Date", "Mood", "Stress", "Sleep", "Notes",
        "Detection Pattern", "Suggestions"
    ])
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
    st.markdown(
        "Welcome! Log your daily wellness, explore interactive charts, and get simple AI insights. "
        "This demo stores data locally in `wellness_data.csv`."
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.metric("📒 Entries", len(data), delta=None)
    with col2:
        if not data.empty:
            avg_mood = data['Mood'].mean()
            avg_sleep = data['Sleep'].mean()
            # Mood metric with emojis
            mood_emoji = "😄" if avg_mood >= 4 else "😐" if avg_mood >= 3 else "😔"
            st.metric("😊 Avg Mood", f"{avg_mood:.2f} / 5 {mood_emoji}", delta=None)
            sleep_emoji = "💤" if avg_sleep >= 7 else "😴"
            st.metric("🛌 Avg Sleep", f"{avg_sleep:.1f} hrs {sleep_emoji}", delta=None)
        else:
            st.info("No data yet — go to 'Daily Log' to add your first entry")
    with col3:
        st.metric("✨ Premium Ideas", "Available", delta=None)

    st.markdown("---")

    # Fun Weekly Snapshot
    st.subheader("📊 Weekly Snapshot")
    if not data.empty:
        last7 = data.tail(7)
        avg_week_mood = last7['Mood'].mean()
        avg_week_sleep = last7['Sleep'].mean()
        st.markdown(f"- Avg Mood this week: **{avg_week_mood:.2f}** {'😄 Keep it up!' if avg_week_mood>=4 else '😐 Could improve'}")
        st.markdown(f"- Avg Sleep this week: **{avg_week_sleep:.1f} hrs** {'💤 Good!' if avg_week_sleep>=7 else '😴 Try to catch up'}")
        st.progress(min(avg_week_mood/5, 1.0))  # Visual bar for mood
        st.progress(min(avg_week_sleep/9, 1.0))  # Visual bar for sleep
    else:
        st.markdown("- Log entries to see weekly insights!")

    st.markdown("---")

    st.subheader("How this helps 🛠️")
    st.markdown("""
- Quick daily logging that takes **30 seconds** ⏱️  
- **Interactive trends** to spot patterns (e.g., low sleep → low mood) 📉📈  
- Simple, interpretable **AI insights** to showcase data science in psychology 🤖  
- Fun **weekly feedback and suggestions** to keep you motivated 🌟
""")

    # Motivational / playful tip
    if not data.empty:
        st.markdown("💡 **Tip of the Day:** Try to add at least one positive note in your journal today! Even small wins count.")


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

        # =========================
        # Mood over time
        # =========================
        st.subheader('Mood over Time')
        st.markdown(
            "This line graph shows how your mood has changed over time. "
            "Tracking mood trends can help you identify patterns, such as days of the week when your mood dips or improves. "
            "Markers indicate individual entries, making it easier to spot sudden changes or improvements."
        )
        fig = px.line(df, x='Date', y='Mood', markers=True, title='Mood Over Time', range_y=[1,5])
        fig.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # Sleep vs Mood scatter
        # =========================
        st.subheader('Sleep vs Mood')
        st.markdown(
            "This scatter plot shows the relationship between your sleep duration and reported mood. "
            "The size of each point represents the amount of exercise you did that day, and the color indicates stress levels. "
            "This allows you to visually assess whether more sleep, less stress, or exercise correlates with higher mood scores."
        )
        fig2 = px.scatter(
            df,
            x='Sleep',
            y='Mood',
            size='Exercise',
            color='Stress',
            hover_data=['Date', 'Journal'],
            title='Sleep vs Mood'
        )
        fig2.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # Screen time & Stress bar (last 14 days)
        # =========================
        st.subheader('Screen Time & Stress (Recent)')
        st.markdown(
            "This grouped bar chart shows your screen time and stress levels over the past 14 days. "
            "It helps you see if high screen time coincides with higher stress, enabling you to identify patterns and adjust habits accordingly."
        )
        recent = df.tail(14)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=recent['Date'].dt.strftime('%b %d'),
            y=recent['ScreenTime'],
            name='Screen Time (hrs)'
        ))
        fig3.add_trace(go.Bar(
            x=recent['Date'].dt.strftime('%b %d'),
            y=recent['Stress'],
            name='Stress (1-10)'
        ))
        fig3.update_layout(
            barmode='group',
            title='Recent Screen Time vs Stress',
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # Correlation heatmap
        # =========================
        st.subheader('Correlation Matrix')
        st.markdown(
            "This heatmap shows the correlations between different tracked measures: mood, sleep, screen time, exercise, stress, and sentiment. "
            "A positive correlation (closer to 1) indicates that the two measures tend to increase together, while a negative correlation (closer to -1) indicates an inverse relationship. "
            "This helps you understand which factors might influence your mood the most."
        )
        corr = df[['Mood', 'Sleep', 'ScreenTime', 'Exercise', 'Stress', 'Sentiment']].corr()
        fig4 = px.imshow(corr, text_auto=True, title='Correlation between Measures', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig4.update_layout(template='plotly_white', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)


# -----------------------------
# AI Insights
# -----------------------------
elif choice == "AI Insights":
    st.header("🤖 AI Insights & Suggestions")

    if data.empty:
        st.info("Add more entries to unlock AI insights.")
    else:
        # Sort data by date
        df = data.sort_values("Date").copy()

        # =========================
        # Generate Detailed AI Insights
        # =========================
        df["Detailed_AI_Insight"] = df.apply(lambda row: generate_detailed_insight(row, df), axis=1)

        # Display latest AI insight
        st.subheader("Latest AI Insight")
        st.markdown(df["Detailed_AI_Insight"].iloc[-1].replace("\n", "  \n"))

        # =========================
        # Pattern-based insights (detailed)
        # =========================
        insights = []

        # Rule: 3 consecutive low mood days
        if len(df) >= 3:
            last3 = df['Mood'].tail(3)
            if all(last3 < 3):
                insights.append((
                    'Low Mood Streak',
                    "You've reported low mood for three consecutive days. Sustained low mood can impact concentration, motivation, and overall well-being. "
                    "It's important to pause and reflect on what may be causing these feelings. Consider talking to someone you trust—a friend, family member, or counselor. "
                    "Sometimes simply expressing your emotions or sharing your experiences can provide relief and clarity. "
                    "Additionally, small daily adjustments like a walk outside, mindful breathing, or journaling can help you break the streak and regain balance."
                ))

        # Sleep correlation
        if df['Sleep'].mean() < 6.5 and df['Mood'].mean() < 3.5:
            insights.append((
                'Sleep & Mood Correlation',
                "Your recent average sleep duration is below 6.5 hours and is accompanied by lower mood scores. Lack of sufficient sleep can exacerbate stress, reduce cognitive performance, "
                "and affect emotional regulation. Establishing a consistent sleep schedule, avoiding screens before bedtime, and creating a calm sleep environment can significantly improve your mood and focus. "
                "Pay attention to sleep quality as well as quantity—sometimes shorter, deep sleep is more restorative than longer, disrupted sleep."
            ))

        # Screen time
        if df['ScreenTime'].mean() > 5 and df['Stress'].mean() > 6:
            insights.append((
                'Screen Time & Stress',
                "High average screen time, especially above 5 hours daily, appears to be associated with elevated stress levels. Prolonged exposure to screens can contribute to eye strain, mental fatigue, and heightened anxiety. "
                "Try to schedule regular breaks from screens, engage in offline hobbies, or practice activities like reading, drawing, or walking outside. "
                "Reducing screen time can help lower stress levels, improve focus, and support better sleep hygiene."
            ))

        # Sentiment vs Mood mismatch
        recent = df.tail(7)
        mismatch = []
        for _, row in recent.iterrows():
            if (row['Sentiment'] < -0.2 and row['Mood'] >= 4) or (row['Sentiment'] > 0.2 and row['Mood'] <= 2):
                mismatch.append((row['Date'], row['Mood'], row['Sentiment']))
        if mismatch:
            insights.append((
                'Sentiment vs Mood Mismatch',
                f"{len(mismatch)} recent entries show a mismatch between your textual sentiment and self-reported mood. "
                "This could suggest underlying emotions not immediately visible in your self-report. Reflecting on these discrepancies can enhance self-awareness, "
                "help you understand your emotional patterns, and improve your ability to communicate feelings effectively. "
                "Consider journaling or speaking aloud about your experiences to bridge the gap between perceived and expressed emotions."
            ))

        # Display pattern-based suggestions
        st.subheader("Detected Patterns / Suggestions")
        if not insights:
            st.success("No concerning patterns detected — keep logging consistently!")
        else:
            for title, text in insights:
                st.warning(f"**{title}**")
                st.write(text)  # Use write for long paragraphs

        # =========================
        # View Past AI Insights by Date (interactive)
        # =========================
        st.subheader("View Past AI Insights")

        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        # Create a dropdown of available dates
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        selected_date = st.selectbox("Select a date to view its AI insight:", dates)

        # Show the insight for the selected date
        insight_to_show = df.loc[df['Date'].dt.strftime('%Y-%m-%d') == selected_date, "Detailed_AI_Insight"].values[0]
        st.markdown(insight_to_show.replace("\n", "  \n"))

        # =========================
        # Mood Prediction Demo
        # =========================
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
    st.markdown(
        "This is a demo of what a **paid tier** could offer. Imagine weekly summaries, personalized tips, and interactive charts to guide your mental well-being!"
    )
    st.markdown("""
**Potential Premium Features**:
- Weekly PDF summaries with charts and AI recommendations
- Personalized AI coach (chat + action plans)
- School/Team dashboards for early intervention (anonymized)
- Export to Google Sheets / Drive sync
""")

    st.markdown('### 🗓 Your Weekly Summary')
    if data.empty:
        st.info('Add data in your Daily Log to see weekly insights.')
    else:
        df = data.sort_values('Date')
        last7 = df.tail(7)

        # 1️⃣ Weekly Stats
        avg_mood = last7['Mood'].mean()
        avg_sleep = last7['Sleep'].mean()
        avg_stress = last7['Stress'].mean()
        total_exercise = last7['Exercise'].sum()

        st.markdown(f"**Mood:** {avg_mood:.1f}/5 {'😄' if avg_mood>=4 else '😐' if avg_mood>=3 else '😔'}")
        st.markdown(f"**Sleep:** {avg_sleep:.1f} hrs/night {'💤 Good!' if avg_sleep>=7 else '😴 Try to improve'}")
        st.markdown(f"**Stress:** {avg_stress:.1f}/10 {'😌 Manageable' if avg_stress<=5 else '😣 Take breaks'}")
        st.markdown(f"**Exercise:** {total_exercise:.0f} mins total this week {'💪 Nice job!' if total_exercise>=150 else '🏃 Keep moving!'}")

        # 2️⃣ Weekly Bar Chart
        st.subheader("Weekly Overview Chart")
        weekly_chart = go.Figure()
        weekly_chart.add_trace(go.Bar(x=last7['Date'].dt.strftime('%a'), y=last7['Mood'], name='Mood', marker_color='mediumseagreen'))
        weekly_chart.add_trace(go.Bar(x=last7['Date'].dt.strftime('%a'), y=last7['Sleep'], name='Sleep', marker_color='royalblue'))
        weekly_chart.add_trace(go.Bar(x=last7['Date'].dt.strftime('%a'), y=last7['Stress'], name='Stress', marker_color='tomato'))
        weekly_chart.update_layout(
            barmode='group',
            title='Mood, Sleep & Stress Over the Last 7 Days',
            yaxis=dict(title='Score / Hours'),
            template='plotly_white'
        )
        st.plotly_chart(weekly_chart, use_container_width=True)

        # 3️⃣ Fun AI Coach Suggestions
        st.subheader("🤖 Weekly AI Coach Recommendations")
        tips = []
        if avg_mood < 3:
            tips.append("Your mood was low this week. Try journaling your thoughts or talking to someone you trust. Small daily wins count!")
        if avg_sleep < 7:
            tips.append("Sleep was slightly below ideal. Prioritize consistent bedtime and screen-free wind-down routines.")
        if avg_stress > 6:
            tips.append("Stress levels were high. Try short mindfulness exercises or walks to reset.")
        if total_exercise < 150:
            tips.append("Aim for at least 30 minutes of activity per day. Even light walks count!")
        if avg_mood >=4 and avg_sleep>=7 and avg_stress<=5 and total_exercise>=150:
            tips.append("Awesome week! 🎉 Keep up your healthy habits!")

        for tip in tips:
            st.info(tip)

        st.markdown('Note: Implementing real premium features would require user authentication and backend storage, but this gives a fun preview!')

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












































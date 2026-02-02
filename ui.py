"""
Bundesliga Betting Predictor - Simple UI
Run with: streamlit run ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from bundesliga_predictor import (
    BundesligaPredictor,
    load_league_data,
    load_bundesliga_data,
    prepare_features_from_raw_data,
    get_latest_team_stats,
    get_team_form_details,
    save_bet_to_csv
)

# Page config
st.set_page_config(
    page_title="Bundesliga Betting Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("⚽ Football Betting Predictor")
st.markdown("**ML + Monte Carlo** predictions for Over/Under 2.5 and BTTS")

# League selector
league_options = {
    "🇩🇪 Bundesliga (Germany)": "D1",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship (England)": "E1",
    "🇹🇷 Super Lig (Turkey)": "T1"
}

selected_league_name = st.selectbox(
    "Select League",
    options=list(league_options.keys()),
    index=0
)
selected_league = league_options[selected_league_name]

# Sidebar for model training
st.sidebar.header("🔧 Model Configuration")
st.sidebar.info(f"**Current League:** {selected_league_name}")

@st.cache_resource
def load_and_train_model(league_code):
    """Load data and train model (cached per league)"""
    from bundesliga_predictor import (
        load_league_data,
        prepare_features_from_raw_data,
        BundesligaPredictor
    )
    
    with st.spinner(f"Loading {league_code} data..."):
        seasons = ['2122', '2223', '2324', '2425', '2526']
        raw_data = load_league_data(league_code, seasons)
    
    with st.spinner("Preparing features..."):
        df, y_over_under, y_btts = prepare_features_from_raw_data(raw_data)
    
    with st.spinner("Training models..."):
        predictor = BundesligaPredictor()
        X = predictor.create_features(df)
        predictor.fit(X, y_over_under, y_btts)
    
    return predictor, raw_data

# Load model for selected league
try:
    predictor, raw_data = load_and_train_model(selected_league)
    st.sidebar.success("✅ Model trained!")
    
    # Get available teams
    all_teams = sorted(set(raw_data['HomeTeam'].unique()) | set(raw_data['AwayTeam'].unique()))
    
    # Show training stats in sidebar
    st.sidebar.metric("Total Matches", len(raw_data))
    st.sidebar.metric("Teams", len(all_teams))
    
    # Team names helper
    with st.sidebar.expander("📋 Team Names"):
        st.caption("Available teams:")
        for team in all_teams:
            st.text(f"• {team}")
    
    # Betting history viewer
    st.sidebar.markdown("---")
    if st.sidebar.button("📊 View Betting History"):
        csv_path = os.path.join(os.getcwd(), 'betting_tracker.csv')
        
        if os.path.isfile(csv_path):
            df_history = pd.read_csv(csv_path)
            
            st.sidebar.success(f"Total bets: {len(df_history)}")
            
            # Show in main area
            with st.expander("📊 Betting History", expanded=True):
                st.dataframe(df_history, use_container_width=True)
                
                # Summary stats
                if len(df_history) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    pending = df_history[df_history['Result'] == 'Pending'].shape[0]
                    won = df_history[df_history['Result'] == 'Won'].shape[0]
                    lost = df_history[df_history['Result'] == 'Lost'].shape[0]
                    total_profit = df_history['Profit_Loss'].sum()
                    
                    col1.metric("Total Bets", len(df_history))
                    col2.metric("Pending", pending)
                    col3.metric("Won", won, delta=f"{lost} lost")
                    col4.metric("Total P/L", f"{total_profit:+.2f} units")
                    
                    if won + lost > 0:
                        st.metric("Win Rate", f"{(won/(won+lost))*100:.1f}%")
        else:
            st.sidebar.info("No bets tracked yet")
    
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Match Details")
    
    # Sort teams alphabetically for easier finding
    all_teams_sorted = sorted(all_teams)
    
    # Set better defaults
    default_home = "Bayern Munich" if "Bayern Munich" in all_teams_sorted else all_teams_sorted[0]
    default_away = "Dortmund" if "Dortmund" in all_teams_sorted else all_teams_sorted[1]
    
    home_team = st.selectbox(
        "Home Team", 
        all_teams_sorted, 
        index=all_teams_sorted.index(default_home)
    )
    
    # Filter away team options to exclude home team
    away_options = [t for t in all_teams_sorted if t != home_team]
    away_default_idx = away_options.index(default_away) if default_away in away_options else 0
    
    away_team = st.selectbox(
        "Away Team", 
        away_options,
        index=away_default_idx
    )

with col2:
    st.subheader("💰 Bookmaker Odds (Decimal)")
    
    tab1, tab2, tab3 = st.tabs(["⚽ Goals", "🚩 Corners", "🟨 Cards"])
    
    with tab1:
        col2a, col2b = st.columns(2)
        with col2a:
            over_odds = st.number_input("Over 2.5", min_value=1.01, max_value=10.0, value=1.85, step=0.01)
            btts_yes_odds = st.number_input("BTTS Yes", min_value=1.01, max_value=10.0, value=1.75, step=0.01)
        with col2b:
            under_odds = st.number_input("Under 2.5", min_value=1.01, max_value=10.0, value=2.00, step=0.01)
            btts_no_odds = st.number_input("BTTS No", min_value=1.01, max_value=10.0, value=2.10, step=0.01)
    
    with tab2:
        col2c, col2d = st.columns(2)
        with col2c:
            over_corners_odds = st.number_input("Over 10.5", min_value=1.01, max_value=10.0, value=1.90, step=0.01, key="over_corners")
        with col2d:
            under_corners_odds = st.number_input("Under 10.5", min_value=1.01, max_value=10.0, value=1.90, step=0.01, key="under_corners")
    
    with tab3:
        col2e, col2f = st.columns(2)
        with col2e:
            over_cards_odds = st.number_input("Over 4.5", min_value=1.01, max_value=10.0, value=1.90, step=0.01, key="over_cards")
        with col2f:
            under_cards_odds = st.number_input("Under 4.5", min_value=1.01, max_value=10.0, value=1.90, step=0.01, key="under_cards")

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    ml_weight = st.slider("ML Model Weight", 0.0, 1.0, 0.6, 0.05)
    mc_weight = 1.0 - ml_weight
    st.write(f"Monte Carlo Weight: {mc_weight:.2f}")
    
    confidence_threshold = st.slider("Confidence Threshold (%)", 50, 80, 60, 5) / 100
    edge_threshold = st.slider("Minimum Edge (%)", 0, 10, 5, 1) / 100

# Initialize session state for predictions
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Predict button
if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
    
    if home_team == away_team:
        st.error("⚠️ Home and away teams must be different!")
        st.stop()
    
    try:
        # Get team stats
        home_stats = get_latest_team_stats(raw_data, home_team, is_home=True, n_games=5)
        away_stats = get_latest_team_stats(raw_data, away_team, is_home=False, n_games=5)
        
        # Get form details
        home_form_details = get_team_form_details(raw_data, home_team, is_home=True, n_games=5)
        away_form_details = get_team_form_details(raw_data, away_team, is_home=False, n_games=5)
        
        # Calculate h2h
        h2h = raw_data[
            ((raw_data['HomeTeam'] == home_team) & (raw_data['AwayTeam'] == away_team)) |
            ((raw_data['HomeTeam'] == away_team) & (raw_data['AwayTeam'] == home_team))
        ]
        h2h_goals_avg = (h2h['FTHG'].mean() + h2h['FTAG'].mean()) if len(h2h) > 0 else 2.5
        
        # Create features
        match_features = pd.DataFrame({
            'home_goals_avg': [home_stats['goals_avg']],
            'away_goals_avg': [away_stats['goals_avg']],
            'home_conceded_avg': [home_stats['conceded_avg']],
            'away_conceded_avg': [away_stats['conceded_avg']],
            'home_form': [home_stats['form']],
            'away_form': [away_stats['form']],
            'h2h_goals_avg': [h2h_goals_avg],
        })
        
        X = predictor.create_features(match_features)
        
        # Get prediction
        prediction = predictor.hybrid_prediction(
            X,
            home_goals_avg=home_stats['goals_avg'],
            away_goals_avg=away_stats['goals_avg'],
            ml_weight=ml_weight,
            mc_weight=mc_weight
        )
        
        # Get corners prediction
        corners_prediction = predictor.predict_corners(
            home_corners_avg=home_stats['corners_avg'],
            away_corners_avg=away_stats['corners_avg']
        )
        
        # Get cards prediction
        cards_prediction = predictor.predict_cards(
            home_yellows_avg=home_stats['yellows_avg'],
            away_yellows_avg=away_stats['yellows_avg'],
            home_reds_avg=home_stats['reds_avg'],
            away_reds_avg=away_stats['reds_avg']
        )
        
        # Display results
        st.markdown("---")
        st.header(f"📊 {home_team} vs {away_team}")
        
        # Form visualization
        st.subheader("📈 Recent Form & Fatigue")
        
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            st.write(f"**{home_team} - Last 5 Games:**")
            
            # Create visual form display
            form_html = "<div style='display: flex; gap: 5px;'>"
            for result in home_form_details['results']:
                if result == 'W':
                    color = '#28a745'  # Green
                    emoji = '✅'
                elif result == 'D':
                    color = '#ffc107'  # Yellow
                    emoji = '➖'
                else:
                    color = '#dc3545'  # Red
                    emoji = '❌'
                
                form_html += f"<div style='background-color: {color}; color: white; padding: 10px 15px; border-radius: 5px; font-weight: bold; text-align: center;'>{emoji} {result}</div>"
            
            form_html += "</div>"
            st.markdown(form_html, unsafe_allow_html=True)
            
            # Days since last match
            if home_form_details['days_since_last_match'] is not None:
                days = home_form_details['days_since_last_match']
                st.metric("Days Since Last Match", f"{days} days")
                
                if days < 3:
                    st.warning("⚠️ Possible fatigue - played recently")
                elif days > 10:
                    st.info("ℹ️ Well rested - long break")
                else:
                    st.success("✅ Normal rest period")
            
            if home_form_details['last_match_date']:
                st.caption(f"Last match: {home_form_details['last_match_date'].strftime('%Y-%m-%d')}")
        
        with col_form2:
            st.write(f"**{away_team} - Last 5 Games:**")
            
            # Create visual form display
            form_html = "<div style='display: flex; gap: 5px;'>"
            for result in away_form_details['results']:
                if result == 'W':
                    color = '#28a745'  # Green
                    emoji = '✅'
                elif result == 'D':
                    color = '#ffc107'  # Yellow
                    emoji = '➖'
                else:
                    color = '#dc3545'  # Red
                    emoji = '❌'
                
                form_html += f"<div style='background-color: {color}; color: white; padding: 10px 15px; border-radius: 5px; font-weight: bold; text-align: center;'>{emoji} {result}</div>"
            
            form_html += "</div>"
            st.markdown(form_html, unsafe_allow_html=True)
            
            # Days since last match
            if away_form_details['days_since_last_match'] is not None:
                days = away_form_details['days_since_last_match']
                st.metric("Days Since Last Match", f"{days} days")
                
                if days < 3:
                    st.warning("⚠️ Possible fatigue - played recently")
                elif days > 10:
                    st.info("ℹ️ Well rested - long break")
                else:
                    st.success("✅ Normal rest period")
            
            if away_form_details['last_match_date']:
                st.caption(f"Last match: {away_form_details['last_match_date'].strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        # Team stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{home_team} Goals/Game", f"{home_stats['goals_avg']:.2f}")
            st.metric(f"{home_team} Corners/Game", f"{home_stats['corners_avg']:.1f}")
            st.metric(f"{home_team} Cards/Game", f"{home_stats['yellows_avg']:.1f}")
            st.metric(f"{home_team} Form", f"{home_stats['form']}/15")
        
        with col2:
            st.metric("Expected Total Goals", f"{prediction['monte_carlo_details']['avg_total_goals']:.2f}")
            st.metric("Expected Total Corners", f"{corners_prediction['avg_total_corners']:.1f}")
            st.metric("Expected Total Cards", f"{cards_prediction['avg_total_cards']:.1f}")
            st.metric("H2H Avg Goals", f"{h2h_goals_avg:.2f}")
        
        with col3:
            st.metric(f"{away_team} Goals/Game", f"{away_stats['goals_avg']:.2f}")
            st.metric(f"{away_team} Corners/Game", f"{away_stats['corners_avg']:.1f}")
            st.metric(f"{away_team} Cards/Game", f"{away_stats['yellows_avg']:.1f}")
            st.metric(f"{away_team} Form", f"{away_stats['form']}/15")
        
        st.markdown("---")
        
        # Predictions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("📈 Over/Under 2.5 Goals")
            
            ou_prob = prediction['over_under']['hybrid_prob']
            ou_prediction = prediction['over_under']['prediction']
            ou_confidence = prediction['over_under']['confidence']
            
            # Calculate EV
            if ou_prediction == "Over 2.5":
                ou_ev = (ou_prob * over_odds) - 1
                ou_bet_odds = over_odds
                ou_bet_type = "Over 2.5"
            else:
                ou_prob_bet = 1 - ou_prob
                ou_ev = (ou_prob_bet * under_odds) - 1
                ou_bet_odds = under_odds
                ou_bet_type = "Under 2.5"
            
            # Display
            st.metric("Prediction", ou_bet_type, f"{ou_confidence:.1%} confidence")
            st.metric("Model Probability", f"{ou_prob:.1%}")
            st.metric("Bookmaker Odds", f"{ou_bet_odds:.2f}", f"Implied: {(1/ou_bet_odds):.1%}")
            st.metric("Expected Value", f"{ou_ev:+.1%}")
            
            # Recommendation
            if ou_ev > edge_threshold and ou_confidence >= confidence_threshold:
                st.success(f"✅ **VALUE BET!** Edge: {ou_ev:.1%}")
            elif ou_ev > 0:
                st.info(f"⚠️ Slight edge: {ou_ev:.1%}")
            else:
                st.error(f"❌ No value: {ou_ev:.1%}")
        
        with col2:
            st.subheader("⚽ Both Teams To Score")
            
            btts_prob = prediction['btts']['hybrid_prob']
            btts_prediction = prediction['btts']['prediction']
            btts_confidence = prediction['btts']['confidence']
            
            # Calculate EV
            if btts_prediction == "Yes":
                btts_ev = (btts_prob * btts_yes_odds) - 1
                btts_bet_odds = btts_yes_odds
                btts_bet_type = "BTTS Yes"
            else:
                btts_prob_bet = 1 - btts_prob
                btts_ev = (btts_prob_bet * btts_no_odds) - 1
                btts_bet_odds = btts_no_odds
                btts_bet_type = "BTTS No"
            
            # Display
            st.metric("Prediction", btts_bet_type, f"{btts_confidence:.1%} confidence")
            st.metric("Model Probability", f"{btts_prob:.1%}")
            st.metric("Bookmaker Odds", f"{btts_bet_odds:.2f}", f"Implied: {(1/btts_bet_odds):.1%}")
            st.metric("Expected Value", f"{btts_ev:+.1%}")
            
            # Recommendation
            if btts_ev > edge_threshold and btts_confidence >= confidence_threshold:
                st.success(f"✅ **VALUE BET!** Edge: {btts_ev:.1%}")
            elif btts_ev > 0:
                st.info(f"⚠️ Slight edge: {btts_ev:.1%}")
            else:
                st.error(f"❌ No value: {btts_ev:.1%}")
        
        with col3:
            st.subheader("🚩 Corners (O/U 10.5)")
            
            corners_prob = corners_prediction['over_10_5']
            
            # Calculate EV for both
            over_corners_ev = (corners_prob * over_corners_odds) - 1
            under_corners_ev = ((1 - corners_prob) * under_corners_odds) - 1
            
            # Determine best bet
            if over_corners_ev > under_corners_ev:
                corners_bet_type = "Over 10.5"
                corners_bet_odds = over_corners_odds
                corners_bet_prob = corners_prob
                corners_ev = over_corners_ev
            else:
                corners_bet_type = "Under 10.5"
                corners_bet_odds = under_corners_odds
                corners_bet_prob = 1 - corners_prob
                corners_ev = under_corners_ev
            
            corners_confidence = max(corners_prob, 1 - corners_prob)
            
            # Display
            st.metric("Prediction", corners_bet_type, f"{corners_confidence:.1%} confidence")
            st.metric("Model Probability", f"{corners_bet_prob:.1%}")
            st.metric("Bookmaker Odds", f"{corners_bet_odds:.2f}", f"Implied: {(1/corners_bet_odds):.1%}")
            st.metric("Expected Value", f"{corners_ev:+.1%}")
            
            # Recommendation
            if corners_ev > edge_threshold and corners_confidence >= confidence_threshold:
                st.success(f"✅ **VALUE BET!** Edge: {corners_ev:.1%}")
            elif corners_ev > 0:
                st.info(f"⚠️ Slight edge: {corners_ev:.1%}")
            else:
                st.error(f"❌ No value: {corners_ev:.1%}")
        
        with col4:
            st.subheader("🟨 Cards (O/U 4.5)")
            
            cards_prob = cards_prediction['over_4_5']
            
            # Calculate EV for both
            over_cards_ev = (cards_prob * over_cards_odds) - 1
            under_cards_ev = ((1 - cards_prob) * under_cards_odds) - 1
            
            # Determine best bet
            if over_cards_ev > under_cards_ev:
                cards_bet_type = "Over 4.5"
                cards_bet_odds = over_cards_odds
                cards_bet_prob = cards_prob
                cards_ev = over_cards_ev
            else:
                cards_bet_type = "Under 4.5"
                cards_bet_odds = under_cards_odds
                cards_bet_prob = 1 - cards_prob
                cards_ev = under_cards_ev
            
            cards_confidence = max(cards_prob, 1 - cards_prob)
            
            # Display
            st.metric("Prediction", cards_bet_type, f"{cards_confidence:.1%} confidence")
            st.metric("Model Probability", f"{cards_bet_prob:.1%}")
            st.metric("Bookmaker Odds", f"{cards_bet_odds:.2f}", f"Implied: {(1/cards_bet_odds):.1%}")
            st.metric("Expected Value", f"{cards_ev:+.1%}")
            
            # Recommendation
            if cards_ev > edge_threshold and cards_confidence >= confidence_threshold:
                st.success(f"✅ **VALUE BET!** Edge: {cards_ev:.1%}")
            elif cards_ev > 0:
                st.info(f"⚠️ Slight edge: {cards_ev:.1%}")
            else:
                st.error(f"❌ No value: {cards_ev:.1%}")
        
        # Monte Carlo details
        st.markdown("---")
        st.subheader("🎲 Monte Carlo Simulation (10,000 iterations)")
        
        goal_dist = prediction['monte_carlo_details']['goal_distribution']
        
        dist_df = pd.DataFrame({
            'Goals': list(goal_dist.keys()),
            'Probability': [v * 100 for v in goal_dist.values()]
        })
        
        st.bar_chart(dist_df.set_index('Goals'))
        
        # Final recommendation
        st.markdown("---")
        st.header("🎯 Final Recommendation")
        
        value_bets = []
        if ou_ev > edge_threshold and ou_confidence >= confidence_threshold:
            value_bets.append(f"**{ou_bet_type}** @ {ou_bet_odds:.2f} (EV: {ou_ev:+.1%}, Confidence: {ou_confidence:.1%})")
        if btts_ev > edge_threshold and btts_confidence >= confidence_threshold:
            value_bets.append(f"**{btts_bet_type}** @ {btts_bet_odds:.2f} (EV: {btts_ev:+.1%}, Confidence: {btts_confidence:.1%})")
        if corners_ev > edge_threshold and corners_confidence >= confidence_threshold:
            value_bets.append(f"**{corners_bet_type} corners** @ {corners_bet_odds:.2f} (EV: {corners_ev:+.1%}, Confidence: {corners_confidence:.1%})")
        if cards_ev > edge_threshold and cards_confidence >= confidence_threshold:
            value_bets.append(f"**{cards_bet_type} cards** @ {cards_bet_odds:.2f} (EV: {cards_ev:+.1%}, Confidence: {cards_confidence:.1%})")
        
        if value_bets:
            st.success("✅ **RECOMMENDED BETS:**")
            for bet in value_bets:
                st.markdown(f"• {bet}")
        else:
            st.warning("⚠️ No value bets found with current settings")
            st.info("Try adjusting odds or threshold settings")
        
        # Store prediction in session state
        st.session_state.prediction_made = True
        st.session_state.last_prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction,
            'corners': corners_prediction,
            'cards': cards_prediction,
            'home_form': home_form_details,
            'away_form': away_form_details
        }
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        import traceback
        st.code(traceback.format_exc())

# Bet tracking section - OUTSIDE the prediction button
if st.session_state.prediction_made:
    st.markdown("---")
    st.subheader("💰 Track This Bet")
    
    # Get teams from session state
    pred_home = st.session_state.last_prediction['home_team']
    pred_away = st.session_state.last_prediction['away_team']
    
    with st.form("bet_tracker_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_date = st.date_input("Match Date", value=pd.Timestamp.now())
            bet_type_input = st.text_input("Bet Type", placeholder="e.g., Over 2.5, BTTS Yes")
        
        with col2:
            bet_odds_input = st.number_input("Odds", min_value=1.01, value=2.0, step=0.01)
            bet_stake_input = st.number_input("Stake (units)", min_value=0.0, value=1.0, step=0.5)
        
        with col3:
            st.metric("Potential Return", f"{bet_stake_input * bet_odds_input:.2f} units")
            st.metric("Potential Profit", f"{bet_stake_input * (bet_odds_input - 1):.2f} units")
        
        submitted = st.form_submit_button("💾 Save Bet", type="primary")
        
        if submitted:
            if not bet_type_input:
                st.error("❌ Please enter a bet type")
            else:
                try:
                    st.write("⏳ Saving bet...")
                    
                    # Save the bet
                    result = save_bet_to_csv(
                        date=match_date.strftime('%Y-%m-%d'),
                        home_team=pred_home,
                        away_team=pred_away,
                        bet_type=bet_type_input,
                        odds=bet_odds_input,
                        stake=bet_stake_input
                    )
                    
                    csv_path = os.path.join(os.getcwd(), 'betting_tracker.csv')
                    
                    # Check if file exists
                    if os.path.exists(csv_path):
                        st.success(f"✅ Bet saved successfully!")
                        st.info(f"📁 File: `{csv_path}`")
                        
                        # Show the saved bet
                        df = pd.read_csv(csv_path)
                        st.write(f"**Total bets: {len(df)}**")
                        st.dataframe(df.tail(3), use_container_width=True)
                    else:
                        st.error(f"❌ FILE NOT FOUND at {csv_path}")
                    
                except Exception as e:
                    st.error(f"❌ Exception: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("⚠️ Bet responsibly. This is for educational purposes only. Past performance does not guarantee future results.")

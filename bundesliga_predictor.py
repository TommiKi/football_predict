"""
Bundesliga Soccer Betting Predictor
Predicts Over/Under 2.5 goals and BTTS (Both Teams To Score)
Uses ML models + Monte Carlo simulation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class BundesligaPredictor:
    """
    Main predictor class combining ML and Monte Carlo methods
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # ML Models
        self.over_under_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
        
        self.btts_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=random_state
        )
        
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from match data
        """
        features = pd.DataFrame()
        
        # Basic stats
        features['home_goals_avg'] = df['home_goals_avg']
        features['away_goals_avg'] = df['away_goals_avg']
        features['home_conceded_avg'] = df['home_conceded_avg']
        features['away_conceded_avg'] = df['away_conceded_avg']
        
        # Form (last 5 games)
        features['home_form'] = df['home_form']
        features['away_form'] = df['away_form']
        
        # Head to head
        features['h2h_goals_avg'] = df['h2h_goals_avg']
        
        # Derived features
        features['total_goals_expected'] = (
            df['home_goals_avg'] + df['away_goals_avg']
        ) / 2
        
        features['home_attack_vs_away_defense'] = (
            df['home_goals_avg'] - df['away_conceded_avg']
        )
        
        features['away_attack_vs_home_defense'] = (
            df['away_goals_avg'] - df['home_conceded_avg']
        )
        
        # Goal scoring probability indicators
        features['home_scoring_prob'] = df['home_goals_avg'] / (df['home_goals_avg'] + 0.01)
        features['away_scoring_prob'] = df['away_goals_avg'] / (df['away_goals_avg'] + 0.01)
        
        return features
    
    def fit(self, X: pd.DataFrame, y_over_under: np.ndarray, y_btts: np.ndarray):
        """
        Train both models
        """
        print("Training models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train over/under model
        print("\n=== Over/Under 2.5 Model ===")
        self.over_under_model.fit(X_scaled, y_over_under)
        ou_score = cross_val_score(self.over_under_model, X_scaled, y_over_under, cv=5, scoring='roc_auc')
        print(f"Cross-val ROC-AUC: {ou_score.mean():.3f} (+/- {ou_score.std():.3f})")
        
        # Train BTTS model
        print("\n=== BTTS Model ===")
        self.btts_model.fit(X_scaled, y_btts)
        btts_score = cross_val_score(self.btts_model, X_scaled, y_btts, cv=5, scoring='roc_auc')
        print(f"Cross-val ROC-AUC: {btts_score.mean():.3f} (+/- {btts_score.std():.3f})")
        
        self.is_fitted = True
        print("\n✓ Models trained successfully!")
        
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get probability predictions from ML models
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted yet. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        return {
            'over_under': self.over_under_model.predict_proba(X_scaled)[:, 1],
            'btts': self.btts_model.predict_proba(X_scaled)[:, 1]
        }
    
    def monte_carlo_simulation(
        self, 
        home_goals_avg: float, 
        away_goals_avg: float,
        n_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation for match outcome
        Uses Poisson distribution for goal scoring
        """
        # Simulate goals using Poisson distribution
        home_goals = np.random.poisson(home_goals_avg, n_simulations)
        away_goals = np.random.poisson(away_goals_avg, n_simulations)
        
        total_goals = home_goals + away_goals
        
        # Calculate probabilities
        over_2_5_prob = np.mean(total_goals > 2.5)
        btts_prob = np.mean((home_goals > 0) & (away_goals > 0))
        
        return {
            'over_2_5': over_2_5_prob,
            'under_2_5': 1 - over_2_5_prob,
            'btts_yes': btts_prob,
            'btts_no': 1 - btts_prob,
            'avg_total_goals': np.mean(total_goals),
            'goal_distribution': {
                '0-1': np.mean(total_goals <= 1),
                '2': np.mean(total_goals == 2),
                '3': np.mean(total_goals == 3),
                '4+': np.mean(total_goals >= 4)
            }
        }
    
    def predict_cards(
        self,
        home_yellows_avg: float,
        away_yellows_avg: float,
        home_reds_avg: float = 0.1,
        away_reds_avg: float = 0.1,
        n_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation for cards prediction
        Uses Poisson distribution for yellow and red cards
        Total booking points: Yellow = 1 point, Red = 2 points
        """
        # Simulate cards using Poisson distribution
        home_yellows = np.random.poisson(home_yellows_avg, n_simulations)
        away_yellows = np.random.poisson(away_yellows_avg, n_simulations)
        home_reds = np.random.poisson(home_reds_avg, n_simulations)
        away_reds = np.random.poisson(away_reds_avg, n_simulations)
        
        total_cards = home_yellows + away_yellows + home_reds + away_reds
        
        # Booking points (often used in betting: Yellow = 10pts, Red = 25pts)
        booking_points = (home_yellows + away_yellows) * 10 + (home_reds + away_reds) * 25
        
        # Common card lines: 3.5, 4.5, 5.5
        return {
            'over_3_5': np.mean(total_cards > 3.5),
            'over_4_5': np.mean(total_cards > 4.5),
            'over_5_5': np.mean(total_cards > 5.5),
            'avg_total_cards': np.mean(total_cards),
            'avg_yellows': np.mean(home_yellows + away_yellows),
            'avg_reds': np.mean(home_reds + away_reds),
            'avg_booking_points': np.mean(booking_points),
            'over_40_booking_pts': np.mean(booking_points > 40),
            'over_50_booking_pts': np.mean(booking_points > 50),
            'over_60_booking_pts': np.mean(booking_points > 60),
            'cards_distribution': {
                '0-3': np.mean(total_cards <= 3),
                '4': np.mean(total_cards == 4),
                '5': np.mean(total_cards == 5),
                '6+': np.mean(total_cards >= 6)
            }
        }
    
    def predict_corners(
        self,
        home_corners_avg: float,
        away_corners_avg: float,
        n_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation for corners prediction
        Uses Poisson distribution for corner kicks
        """
        # Simulate corners using Poisson distribution
        home_corners = np.random.poisson(home_corners_avg, n_simulations)
        away_corners = np.random.poisson(away_corners_avg, n_simulations)
        
        total_corners = home_corners + away_corners
        
        # Common corner lines: 8.5, 9.5, 10.5, 11.5
        return {
            'over_8_5': np.mean(total_corners > 8.5),
            'over_9_5': np.mean(total_corners > 9.5),
            'over_10_5': np.mean(total_corners > 10.5),
            'over_11_5': np.mean(total_corners > 11.5),
            'avg_total_corners': np.mean(total_corners),
            'corners_distribution': {
                '0-7': np.mean(total_corners <= 7),
                '8-9': np.mean((total_corners >= 8) & (total_corners <= 9)),
                '10-11': np.mean((total_corners >= 10) & (total_corners <= 11)),
                '12-13': np.mean((total_corners >= 12) & (total_corners <= 13)),
                '14+': np.mean(total_corners >= 14)
            }
        }
    
    def hybrid_prediction(
        self,
        X: pd.DataFrame,
        home_goals_avg: float,
        away_goals_avg: float,
        ml_weight: float = 0.6,
        mc_weight: float = 0.4
    ) -> Dict[str, any]:
        """
        Combine ML and Monte Carlo predictions
        """
        # Get ML predictions
        ml_probs = self.predict_proba(X)
        
        # Get Monte Carlo predictions
        mc_results = self.monte_carlo_simulation(home_goals_avg, away_goals_avg)
        
        # Weighted combination
        hybrid_over = (ml_weight * ml_probs['over_under'][0] + 
                       mc_weight * mc_results['over_2_5'])
        
        hybrid_btts = (ml_weight * ml_probs['btts'][0] + 
                       mc_weight * mc_results['btts_yes'])
        
        return {
            'over_under': {
                'ml_prob': ml_probs['over_under'][0],
                'mc_prob': mc_results['over_2_5'],
                'hybrid_prob': hybrid_over,
                'prediction': 'Over 2.5' if hybrid_over > 0.5 else 'Under 2.5',
                'confidence': max(hybrid_over, 1 - hybrid_over)
            },
            'btts': {
                'ml_prob': ml_probs['btts'][0],
                'mc_prob': mc_results['btts_yes'],
                'hybrid_prob': hybrid_btts,
                'prediction': 'Yes' if hybrid_btts > 0.5 else 'No',
                'confidence': max(hybrid_btts, 1 - hybrid_btts)
            },
            'monte_carlo_details': mc_results
        }


def load_league_data(league_code: str, seasons: List[str] = None, cache_dir: str = './data_cache') -> pd.DataFrame:
    """
    Load league data from football-data.co.uk
    Downloads and caches CSV files locally to avoid repeated downloads
    
    League codes:
    - D1: Bundesliga (Germany)
    - E1: Championship (England)
    - T1: Super Lig (Turkey)
    """
    import os
    
    if seasons is None:
        seasons = ['2122', '2223', '2324', '2425', '2526']
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    all_data = []
    base_url = f"https://www.football-data.co.uk/mmz4281/{{}}/{league_code}.csv"
    
    league_names = {
        'D1': 'Bundesliga',
        'E1': 'Championship',
        'T1': 'Super Lig'
    }
    league_name = league_names.get(league_code, league_code)
    
    print(f"\n📥 Loading {league_name} data for {len(seasons)} seasons...")
    
    for season in seasons:
        cache_file = os.path.join(cache_dir, f'{league_code}_{season}.csv')
        
        # Always re-download current season to get latest matches
        is_current_season = (season == '2526')
        
        # Try to load from cache first (unless current season)
        if os.path.exists(cache_file) and not is_current_season:
            try:
                df = pd.read_csv(cache_file, encoding='latin1')
                df['Season'] = season
                df['League'] = league_code
                all_data.append(df)
                print(f"  ✓ Season {season[:2]}/{season[2:]}: {len(df)} matches (from cache)")
                continue
            except Exception as e:
                print(f"  ⚠️  Cache file corrupted for {season}, re-downloading...")
        
        # Download if not in cache or if current season
        try:
            url = base_url.format(season)
            df = pd.read_csv(url, encoding='latin1')
            df['Season'] = season
            df['League'] = league_code
            
            # Save to cache
            df.to_csv(cache_file, index=False, encoding='latin1')
            
            all_data.append(df)
            status = "downloaded & cached (current season)" if is_current_season else "downloaded & cached"
            print(f"  ✓ Season {season[:2]}/{season[2:]}: {len(df)} matches ({status})")
        except Exception as e:
            print(f"  ✗ Failed to load season {season}: {e}")
    
    if not all_data:
        raise ValueError("No data loaded successfully")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total matches loaded: {len(combined_df)}")
    print(f"💾 Data cached in: {os.path.abspath(cache_dir)}")
    
    return combined_df


def load_bundesliga_data(seasons: List[str] = None, cache_dir: str = './data_cache') -> pd.DataFrame:
    """
    Load Bundesliga data (wrapper for backward compatibility)
    """
    return load_league_data('D1', seasons, cache_dir)


def clear_cache(cache_dir: str = './data_cache'):
    """
    Clear all cached data files
    """
    import os
    import shutil
    
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"✓ Cache cleared: {cache_dir}")
    else:
        print(f"No cache found at: {cache_dir}")


def get_team_form_details(raw_df: pd.DataFrame, team: str, is_home: bool = None, n_games: int = 5) -> Dict:
    """
    Get detailed form history including results and dates for last N games
    Gets ALL matches for the team (both home and away) to show true recent form
    is_home parameter is kept for compatibility but not used - we want all matches
    """
    # Get ALL matches where team played (home or away)
    team_matches = raw_df[
        (raw_df['HomeTeam'] == team) | (raw_df['AwayTeam'] == team)
    ].copy()
    
    # CRITICAL: Ensure Date is datetime before sorting
    if team_matches['Date'].dtype == 'object':  # String type
        team_matches['Date'] = pd.to_datetime(team_matches['Date'], format='%d/%m/%Y', errors='coerce')
    
    team_matches = team_matches.sort_values('Date')
    
    if len(team_matches) < 1:
        return {
            'results': [],
            'dates': [],
            'last_match_date': None,
            'days_since_last_match': None
        }
    
    recent = team_matches.tail(n_games)
    
    results = []
    dates = []
    
    for _, match in recent.iterrows():
        # Determine if team was home or away in this match
        was_home = match['HomeTeam'] == team
        
        # Determine result from team's perspective
        if was_home:
            if match['FTR'] == 'H':
                result = 'W'
            elif match['FTR'] == 'D':
                result = 'D'
            else:
                result = 'L'
        else:  # was away
            if match['FTR'] == 'A':
                result = 'W'
            elif match['FTR'] == 'D':
                result = 'D'
            else:
                result = 'L'
        
        results.append(result)
        dates.append(match['Date'])
    
    # Calculate days since last match
    last_match_date = dates[-1] if dates else None
    days_since_last = None
    
    if last_match_date:
        from datetime import datetime
        if isinstance(last_match_date, str):
            last_match_date = pd.to_datetime(last_match_date)
        days_since_last = (pd.Timestamp.now() - last_match_date).days
    
    return {
        'results': list(reversed(results)),  # Reverse to show newest first
        'dates': list(reversed(dates)),
        'last_match_date': last_match_date,
        'days_since_last_match': days_since_last
    }


def get_latest_team_stats(raw_df: pd.DataFrame, team: str, is_home: bool, n_games: int = 5) -> Dict:
    """
    Get the most recent stats for a team based on all available data
    """
    if is_home:
        team_matches = raw_df[raw_df['HomeTeam'] == team].copy()
        goals_for = 'FTHG'
        goals_against = 'FTAG'
        corners_for = 'HC'
        corners_against = 'AC'
        yellows = 'HY'
        reds = 'HR'
    else:
        team_matches = raw_df[raw_df['AwayTeam'] == team].copy()
        goals_for = 'FTAG'
        goals_against = 'FTHG'
        corners_for = 'AC'
        corners_against = 'HC'
        yellows = 'AY'
        reds = 'AR'
    
    team_matches = team_matches.sort_values('Date')
    
    if len(team_matches) < n_games:
        print(f"  ⚠️  Warning: Only {len(team_matches)} matches found for {team}")
        return {
            'goals_avg': 1.5,
            'conceded_avg': 1.5,
            'form': 7.5,
            'corners_avg': 5.0,
            'corners_against_avg': 5.0,
            'yellows_avg': 1.5,
            'reds_avg': 0.1
        }
    
    recent = team_matches.tail(n_games)
    
    # Calculate form
    if is_home:
        wins = (recent['FTR'] == 'H').sum()
        draws = (recent['FTR'] == 'D').sum()
    else:
        wins = (recent['FTR'] == 'A').sum()
        draws = (recent['FTR'] == 'D').sum()
    
    form = wins * 3 + draws * 1
    
    # Calculate corners (if available)
    corners_avg = recent[corners_for].mean() if corners_for in recent.columns and recent[corners_for].notna().any() else 5.0
    corners_against_avg = recent[corners_against].mean() if corners_against in recent.columns and recent[corners_against].notna().any() else 5.0
    
    # Calculate cards (if available)
    yellows_avg = recent[yellows].mean() if yellows in recent.columns and recent[yellows].notna().any() else 1.5
    reds_avg = recent[reds].mean() if reds in recent.columns and recent[reds].notna().any() else 0.1
    
    return {
        'goals_avg': recent[goals_for].mean(),
        'conceded_avg': recent[goals_against].mean(),
        'form': form,
        'recent_matches': len(recent),
        'corners_avg': corners_avg,
        'corners_against_avg': corners_against_avg,
        'yellows_avg': yellows_avg,
        'reds_avg': reds_avg
    }


def save_bet_to_csv(
    date: str,
    home_team: str,
    away_team: str,
    bet_type: str,
    odds: float,
    stake: float = 0,
    csv_file: str = None
):
    """
    Save a bet to CSV for tracking
    """
    import os
    from datetime import datetime
    
    # Use absolute path based on current working directory
    if csv_file is None:
        csv_file = os.path.join(os.getcwd(), 'betting_tracker.csv')
    
    bet_data = {
        'Date': date,
        'Match': f"{home_team} vs {away_team}",
        'Home_Team': home_team,
        'Away_Team': away_team,
        'Bet_Type': bet_type,
        'Odds': odds,
        'Stake': stake,
        'Potential_Return': stake * odds if stake > 0 else 0,
        'Result': 'Pending',
        'Profit_Loss': 0,
        'Logged_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check if file exists
    file_exists = os.path.isfile(csv_file)
    
    # Create DataFrame
    df_new = pd.DataFrame([bet_data])
    
    # Append to CSV
    if file_exists:
        df_new.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_file, mode='w', header=True, index=False)
    
    print(f"\n✅ Bet saved to {csv_file}")
    return bet_data


def view_betting_history(csv_file: str = 'betting_tracker.csv'):
    """
    View all bets from the tracking CSV
    """
    import os
    
    if not os.path.isfile(csv_file):
        print(f"\n📊 No betting history found at {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    
    print(f"\n📊 BETTING HISTORY ({len(df)} bets)")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Summary stats
    if len(df) > 0:
        pending = df[df['Result'] == 'Pending'].shape[0]
        won = df[df['Result'] == 'Won'].shape[0]
        lost = df[df['Result'] == 'Lost'].shape[0]
        total_profit = df['Profit_Loss'].sum()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Bets: {len(df)}")
        print(f"Pending: {pending} | Won: {won} | Lost: {lost}")
        if won + lost > 0:
            print(f"Win Rate: {(won/(won+lost))*100:.1f}%")
        print(f"Total P/L: {total_profit:+.2f} units")
    
    return df


def update_bet_result(
    csv_file: str,
    bet_index: int,
    result: str,
    actual_return: float = None
):
    """
    Update a bet's result (Won/Lost)
    """
    import os
    
    if not os.path.isfile(csv_file):
        print(f"❌ File {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    if bet_index >= len(df):
        print(f"❌ Bet index {bet_index} not found")
        return
    
    df.loc[bet_index, 'Result'] = result
    
    if result == 'Won':
        if actual_return is not None:
            df.loc[bet_index, 'Profit_Loss'] = actual_return - df.loc[bet_index, 'Stake']
        else:
            df.loc[bet_index, 'Profit_Loss'] = (df.loc[bet_index, 'Stake'] * df.loc[bet_index, 'Odds']) - df.loc[bet_index, 'Stake']
    elif result == 'Lost':
        df.loc[bet_index, 'Profit_Loss'] = -df.loc[bet_index, 'Stake']
    
    df.to_csv(csv_file, index=False)
    print(f"✅ Bet {bet_index} updated to: {result}")


def predict_upcoming_match(
    predictor: BundesligaPredictor,
    raw_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    ml_weight: float = 0.6,
    mc_weight: float = 0.4
) -> Dict:
    """
    Predict an upcoming match using latest team stats
    """
    print(f"\n{'='*60}")
    print(f"🔮 PREDICTION: {home_team} vs {away_team}")
    print(f"{'='*60}")
    
    # Get latest stats for both teams
    home_stats = get_latest_team_stats(raw_df, home_team, is_home=True, n_games=5)
    away_stats = get_latest_team_stats(raw_df, away_team, is_home=False, n_games=5)
    
    # Get form details
    home_form = get_team_form_details(raw_df, home_team, is_home=True, n_games=5)
    away_form = get_team_form_details(raw_df, away_team, is_home=False, n_games=5)
    
    # Display form visualization
    print(f"\n📈 RECENT FORM & FATIGUE")
    print(f"\n  {home_team} - Last 5 games:")
    form_str = "    "
    for result in home_form['results']:
        if result == 'W':
            form_str += "✅ W  "
        elif result == 'D':
            form_str += "➖ D  "
        else:
            form_str += "❌ L  "
    print(form_str)
    
    if home_form['days_since_last_match'] is not None:
        days = home_form['days_since_last_match']
        print(f"    Last match: {days} days ago", end="")
        if days < 3:
            print(" ⚠️  (Possible fatigue)")
        elif days > 10:
            print(" ℹ️  (Well rested)")
        else:
            print(" ✅ (Normal rest)")
    
    print(f"\n  {away_team} - Last 5 games:")
    form_str = "    "
    for result in away_form['results']:
        if result == 'W':
            form_str += "✅ W  "
        elif result == 'D':
            form_str += "➖ D  "
        else:
            form_str += "❌ L  "
    print(form_str)
    
    if away_form['days_since_last_match'] is not None:
        days = away_form['days_since_last_match']
        print(f"    Last match: {days} days ago", end="")
        if days < 3:
            print(" ⚠️  (Possible fatigue)")
        elif days > 10:
            print(" ℹ️  (Well rested)")
        else:
            print(" ✅ (Normal rest)")
    
    # Calculate head to head
    h2h = raw_df[
        ((raw_df['HomeTeam'] == home_team) & (raw_df['AwayTeam'] == away_team)) |
        ((raw_df['HomeTeam'] == away_team) & (raw_df['AwayTeam'] == home_team))
    ]
    h2h_goals_avg = (h2h['FTHG'].mean() + h2h['FTAG'].mean()) if len(h2h) > 0 else 2.5
    
    # Create feature dataframe
    match_features = pd.DataFrame({
        'home_goals_avg': [home_stats['goals_avg']],
        'away_goals_avg': [away_stats['goals_avg']],
        'home_conceded_avg': [home_stats['conceded_avg']],
        'away_conceded_avg': [away_stats['conceded_avg']],
        'home_form': [home_stats['form']],
        'away_form': [away_stats['form']],
        'h2h_goals_avg': [h2h_goals_avg],
    })
    
    # Engineer features
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
    
    # Display team stats
    print(f"\n📊 Team Statistics (Last 5 games):")
    print(f"\n  {home_team} (Home):")
    print(f"    Goals/game: {home_stats['goals_avg']:.2f}")
    print(f"    Conceded/game: {home_stats['conceded_avg']:.2f}")
    print(f"    Corners/game: {home_stats['corners_avg']:.1f}")
    print(f"    Yellow cards/game: {home_stats['yellows_avg']:.1f}")
    print(f"    Form: {home_stats['form']}/15 points")
    
    print(f"\n  {away_team} (Away):")
    print(f"    Goals/game: {away_stats['goals_avg']:.2f}")
    print(f"    Conceded/game: {away_stats['conceded_avg']:.2f}")
    print(f"    Corners/game: {away_stats['corners_avg']:.1f}")
    print(f"    Yellow cards/game: {away_stats['yellows_avg']:.1f}")
    print(f"    Form: {away_stats['form']}/15 points")
    
    print(f"\n  Head-to-Head: {len(h2h)} matches, avg {h2h_goals_avg:.2f} goals")
    
    # Display predictions
    print(f"\n{'='*60}")
    print("🎯 PREDICTIONS")
    print(f"{'='*60}")
    
    print(f"\n📈 OVER/UNDER 2.5 GOALS")
    print(f"  Prediction: {prediction['over_under']['prediction']}")
    print(f"  Confidence: {prediction['over_under']['confidence']:.1%}")
    print(f"  Probability: {prediction['over_under']['hybrid_prob']:.1%}")
    print(f"  (ML: {prediction['over_under']['ml_prob']:.1%} | MC: {prediction['over_under']['mc_prob']:.1%})")
    
    print(f"\n⚽ BOTH TEAMS TO SCORE")
    print(f"  Prediction: {prediction['btts']['prediction']}")
    print(f"  Confidence: {prediction['btts']['confidence']:.1%}")
    print(f"  Probability: {prediction['btts']['hybrid_prob']:.1%}")
    print(f"  (ML: {prediction['btts']['ml_prob']:.1%} | MC: {prediction['btts']['mc_prob']:.1%})")
    
    print(f"\n🎲 MONTE CARLO INSIGHTS")
    print(f"  Expected Total Goals: {prediction['monte_carlo_details']['avg_total_goals']:.2f}")
    print(f"  Goal Distribution:")
    for range_label, prob in prediction['monte_carlo_details']['goal_distribution'].items():
        bar = '█' * int(prob * 30)
        print(f"    {range_label:3} goals: {bar} {prob:.1%}")
    
    # Display corners prediction
    print(f"\n🚩 CORNERS PREDICTION")
    print(f"  Expected Total Corners: {corners_prediction['avg_total_corners']:.1f}")
    print(f"  Probabilities:")
    print(f"    Over 8.5:  {corners_prediction['over_8_5']:.1%}")
    print(f"    Over 9.5:  {corners_prediction['over_9_5']:.1%}")
    print(f"    Over 10.5: {corners_prediction['over_10_5']:.1%} ⭐")
    print(f"    Over 11.5: {corners_prediction['over_11_5']:.1%}")
    
    # Corners recommendation
    if corners_prediction['over_10_5'] > 0.60:
        print(f"  → Prediction: Over 10.5 corners (Confidence: {corners_prediction['over_10_5']:.1%})")
    elif corners_prediction['over_10_5'] < 0.40:
        print(f"  → Prediction: Under 10.5 corners (Confidence: {(1-corners_prediction['over_10_5']):.1%})")
    else:
        print(f"  → Prediction: Close call, consider skipping corners market")
    
    # Display cards prediction
    print(f"\n🟨 CARDS PREDICTION")
    print(f"  Expected Total Cards: {cards_prediction['avg_total_cards']:.1f}")
    print(f"  Expected Yellow Cards: {cards_prediction['avg_yellows']:.1f}")
    print(f"  Expected Red Cards: {cards_prediction['avg_reds']:.2f}")
    print(f"  Expected Booking Points: {cards_prediction['avg_booking_points']:.0f}")
    print(f"\n  Total Cards Probabilities:")
    print(f"    Over 3.5: {cards_prediction['over_3_5']:.1%}")
    print(f"    Over 4.5: {cards_prediction['over_4_5']:.1%} ⭐")
    print(f"    Over 5.5: {cards_prediction['over_5_5']:.1%}")
    print(f"\n  Booking Points Probabilities:")
    print(f"    Over 40pts: {cards_prediction['over_40_booking_pts']:.1%}")
    print(f"    Over 50pts: {cards_prediction['over_50_booking_pts']:.1%}")
    print(f"    Over 60pts: {cards_prediction['over_60_booking_pts']:.1%}")
    
    # Cards recommendation
    if cards_prediction['over_4_5'] > 0.60:
        print(f"  → Prediction: Over 4.5 cards (Confidence: {cards_prediction['over_4_5']:.1%})")
    elif cards_prediction['over_4_5'] < 0.40:
        print(f"  → Prediction: Under 4.5 cards (Confidence: {(1-cards_prediction['over_4_5']):.1%})")
    else:
        print(f"  → Prediction: Close call, consider skipping cards market")
    
    # Betting recommendation
    print(f"\n💡 BETTING RECOMMENDATION")
    ou_conf = prediction['over_under']['confidence']
    btts_conf = prediction['btts']['confidence']
    
    if ou_conf >= 0.65 and btts_conf >= 0.65:
        print(f"  ✅ STRONG: Both markets show high confidence")
        print(f"     → {prediction['over_under']['prediction']} @ {ou_conf:.1%}")
        print(f"     → BTTS {prediction['btts']['prediction']} @ {btts_conf:.1%}")
    elif ou_conf >= 0.65:
        print(f"  ✅ BET: {prediction['over_under']['prediction']} (confidence: {ou_conf:.1%})")
        print(f"  ⚠️  SKIP: BTTS (confidence: {btts_conf:.1%})")
    elif btts_conf >= 0.65:
        print(f"  ⚠️  SKIP: Over/Under (confidence: {ou_conf:.1%})")
        print(f"  ✅ BET: BTTS {prediction['btts']['prediction']} (confidence: {btts_conf:.1%})")
    else:
        print(f"  ⚠️  SKIP: Low confidence on both markets")
        print(f"     Over/Under: {ou_conf:.1%} | BTTS: {btts_conf:.1%}")
    
    print(f"\n{'='*60}\n")
    
    return prediction


def predict_multiple_matches(
    predictor: BundesligaPredictor,
    raw_df: pd.DataFrame,
    matches: List[Tuple[str, str]],
    confidence_threshold: float = 0.65
) -> pd.DataFrame:
    """
    Predict multiple upcoming matches and return results as DataFrame
    """
    results = []
    
    for home_team, away_team in matches:
        prediction = predict_upcoming_match(predictor, raw_df, home_team, away_team)
        
        results.append({
            'Match': f"{home_team} vs {away_team}",
            'OU_Prediction': prediction['over_under']['prediction'],
            'OU_Confidence': prediction['over_under']['confidence'],
            'OU_Prob': prediction['over_under']['hybrid_prob'],
            'BTTS_Prediction': prediction['btts']['prediction'],
            'BTTS_Confidence': prediction['btts']['confidence'],
            'BTTS_Prob': prediction['btts']['hybrid_prob'],
            'Expected_Goals': prediction['monte_carlo_details']['avg_total_goals'],
            'Recommend_OU': '✅' if prediction['over_under']['confidence'] >= confidence_threshold else '❌',
            'Recommend_BTTS': '✅' if prediction['btts']['confidence'] >= confidence_threshold else '❌',
        })
    
    df = pd.DataFrame(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 BETTING SUMMARY")
    print(f"{'='*60}")
    print(f"\nRecommended Over/Under bets: {df['Recommend_OU'].value_counts().get('✅', 0)}/{len(df)}")
    print(f"Recommended BTTS bets: {df['Recommend_BTTS'].value_counts().get('✅', 0)}/{len(df)}")
    
    return df
    """
    Calculate rolling statistics for a team
    """
    if is_home:
        team_matches = df[df['HomeTeam'] == team].copy()
        goals_for = 'FTHG'
        goals_against = 'FTAG'
    else:
        team_matches = df[df['AwayTeam'] == team].copy()
        goals_for = 'FTAG'
        goals_against = 'FTHG'
    
    team_matches = team_matches.sort_values('Date')
    
    if len(team_matches) < n_games:
        return {
            'goals_avg': 1.5,
            'conceded_avg': 1.5,
            'form': 7.5
        }
    
    recent = team_matches.tail(n_games)
    
    # Calculate form (3 points win, 1 draw, 0 loss)
    if is_home:
        wins = (recent['FTR'] == 'H').sum()
        draws = (recent['FTR'] == 'D').sum()
    else:
        wins = (recent['FTR'] == 'A').sum()
        draws = (recent['FTR'] == 'D').sum()
    
    form = wins * 3 + draws * 1
    
    return {
        'goals_avg': recent[goals_for].mean(),
        'conceded_avg': recent[goals_against].mean(),
        'form': form
    }


def prepare_features_from_raw_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare features and targets from raw Bundesliga data
    """
    print("\n🔧 Engineering features from historical data...")
    
    # Ensure Date is datetime
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='%d/%m/%Y', errors='coerce')
    raw_df = raw_df.sort_values('Date').reset_index(drop=True)
    
    features_list = []
    
    for idx, row in raw_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing match {idx}/{len(raw_df)}...", end='\r')
        
        # Get historical data before this match
        historical = raw_df.iloc[:idx]
        
        if len(historical) < 10:  # Skip if not enough history
            continue
        
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        # Get team stats
        home_stats = get_latest_team_stats(historical, home_team, is_home=True, n_games=5)
        away_stats = get_latest_team_stats(historical, away_team, is_home=False, n_games=5)
        
        # Get form details for visualization
        home_form = get_team_form_details(historical, home_team, is_home=True, n_games=5)
        away_form = get_team_form_details(historical, away_team, is_home=False, n_games=5)
        
        # Head to head
        h2h = historical[
            ((historical['HomeTeam'] == home_team) & (historical['AwayTeam'] == away_team)) |
            ((historical['HomeTeam'] == away_team) & (historical['AwayTeam'] == home_team))
        ]
        h2h_goals_avg = (h2h['FTHG'].mean() + h2h['FTAG'].mean()) if len(h2h) > 0 else 2.5
        
        features = {
            'home_goals_avg': home_stats['goals_avg'],
            'away_goals_avg': away_stats['goals_avg'],
            'home_conceded_avg': home_stats['conceded_avg'],
            'away_conceded_avg': away_stats['conceded_avg'],
            'home_form': home_stats['form'],
            'away_form': away_stats['form'],
            'h2h_goals_avg': h2h_goals_avg,
            'actual_home_goals': row['FTHG'],
            'actual_away_goals': row['FTAG'],
        }
        
        features_list.append(features)
    
    print(f"\n✓ Created {len(features_list)} feature sets")
    
    df = pd.DataFrame(features_list)
    
    # Create targets
    total_goals = df['actual_home_goals'] + df['actual_away_goals']
    y_over_under = (total_goals > 2.5).astype(int)
    y_btts = ((df['actual_home_goals'] > 0) & (df['actual_away_goals'] > 0)).astype(int)
    
    # Drop actual goals from features
    df = df.drop(['actual_home_goals', 'actual_away_goals'], axis=1)
    
    print(f"  Over 2.5 rate: {y_over_under.mean():.1%}")
    print(f"  BTTS rate: {y_btts.mean():.1%}")
    
    return df, y_over_under, y_btts


def backtest_strategy(
    predictor: BundesligaPredictor,
    X_test: pd.DataFrame,
    y_over_under_test: np.ndarray,
    y_btts_test: np.ndarray,
    confidence_threshold: float = 0.65
) -> Dict[str, any]:
    """
    Backtest the betting strategy
    """
    predictions = predictor.predict_proba(X_test)
    
    # Over/Under strategy
    ou_confident_bets = predictions['over_under'] > confidence_threshold
    ou_correct = (predictions['over_under'] > 0.5) == y_over_under_test
    
    # BTTS strategy
    btts_confident_bets = predictions['btts'] > confidence_threshold
    btts_correct = (predictions['btts'] > 0.5) == y_btts_test
    
    results = {
        'over_under': {
            'total_bets': np.sum(ou_confident_bets),
            'correct_bets': np.sum(ou_correct[ou_confident_bets]),
            'accuracy': np.mean(ou_correct[ou_confident_bets]) if np.sum(ou_confident_bets) > 0 else 0,
            'roi_estimate': calculate_roi(ou_correct[ou_confident_bets]) if np.sum(ou_confident_bets) > 0 else 0
        },
        'btts': {
            'total_bets': np.sum(btts_confident_bets),
            'correct_bets': np.sum(btts_correct[btts_confident_bets]),
            'accuracy': np.mean(btts_correct[btts_confident_bets]) if np.sum(btts_confident_bets) > 0 else 0,
            'roi_estimate': calculate_roi(btts_correct[btts_confident_bets]) if np.sum(btts_confident_bets) > 0 else 0
        }
    }
    
    return results


def calculate_roi(correct_predictions: np.ndarray, odds: float = 1.9) -> float:
    """
    Calculate ROI assuming average odds
    """
    total_stake = len(correct_predictions)
    if total_stake == 0:
        return 0
    
    winnings = np.sum(correct_predictions) * odds
    roi = ((winnings - total_stake) / total_stake) * 100
    return roi


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FOOTBALL BETTING PREDICTOR")
    print("Over/Under 2.5 + BTTS Predictions")
    print("=" * 60)
    
    # Select league
    print("\nAvailable leagues:")
    print("1. D1 - Bundesliga (Germany)")
    print("2. E1 - Championship (England)")
    print("3. T1 - Super Lig (Turkey)")
    
    league_choice = input("\nSelect league (1-3, or press Enter for Bundesliga): ").strip()
    
    league_map = {
        '1': 'D1',
        '2': 'E1',
        '3': 'T1',
        '': 'D1'  # default
    }
    
    league_code = league_map.get(league_choice, 'D1')
    league_names = {
        'D1': 'Bundesliga',
        'E1': 'Championship',
        'T1': 'Super Lig'
    }
    
    print(f"\n✓ Selected: {league_names[league_code]}")
    
    # Load league data (including current season)
    seasons = ['2122', '2223', '2324', '2425', '2526']
    raw_data = load_league_data(league_code, seasons)
    
    # Prepare features and targets
    df, y_over_under, y_btts = prepare_features_from_raw_data(raw_data)
    
    # Create features
    predictor = BundesligaPredictor()
    X = predictor.create_features(df)
    
    # Split data
    X_train, X_test, y_ou_train, y_ou_test, y_btts_train, y_btts_test = train_test_split(
        X, y_over_under, y_btts, test_size=0.2, random_state=42
    )
    
    # Train models
    predictor.fit(X_train, y_ou_train, y_btts_train)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION: Bayern Munich vs Borussia Dortmund")
    print("=" * 60)
    
    # Sample match features
    sample_match = pd.DataFrame({
        'home_goals_avg': [2.1],
        'away_goals_avg': [1.8],
        'home_conceded_avg': [0.9],
        'away_conceded_avg': [1.2],
        'home_form': [12],
        'away_form': [10],
        'h2h_goals_avg': [3.2],
    })
    
    sample_features = predictor.create_features(sample_match)
    
    # Get hybrid prediction
    prediction = predictor.hybrid_prediction(
        sample_features,
        home_goals_avg=2.1,
        away_goals_avg=1.8,
        ml_weight=0.6,
        mc_weight=0.4
    )
    
    print("\n📈 OVER/UNDER 2.5 GOALS")
    print(f"  ML Probability: {prediction['over_under']['ml_prob']:.1%}")
    print(f"  Monte Carlo Probability: {prediction['over_under']['mc_prob']:.1%}")
    print(f"  Hybrid Probability: {prediction['over_under']['hybrid_prob']:.1%}")
    print(f"  → Prediction: {prediction['over_under']['prediction']}")
    print(f"  → Confidence: {prediction['over_under']['confidence']:.1%}")
    
    print("\n⚽ BOTH TEAMS TO SCORE")
    print(f"  ML Probability: {prediction['btts']['ml_prob']:.1%}")
    print(f"  Monte Carlo Probability: {prediction['btts']['mc_prob']:.1%}")
    print(f"  Hybrid Probability: {prediction['btts']['hybrid_prob']:.1%}")
    print(f"  → Prediction: {prediction['btts']['prediction']}")
    print(f"  → Confidence: {prediction['btts']['confidence']:.1%}")
    
    print("\n🎲 MONTE CARLO SIMULATION DETAILS")
    print(f"  Expected Total Goals: {prediction['monte_carlo_details']['avg_total_goals']:.2f}")
    print(f"  Goal Distribution:")
    for range_label, prob in prediction['monte_carlo_details']['goal_distribution'].items():
        print(f"    {range_label} goals: {prob:.1%}")
    
    # Backtest
    print("\n" + "=" * 60)
    print("BACKTESTING RESULTS (Test Set)")
    print("=" * 60)
    
    backtest_results = backtest_strategy(
        predictor,
        X_test,
        y_ou_test,
        y_btts_test,
        confidence_threshold=0.60
    )
    
    print("\n📊 Over/Under 2.5 Strategy:")
    print(f"  Total Bets: {backtest_results['over_under']['total_bets']}")
    print(f"  Correct: {backtest_results['over_under']['correct_bets']}")
    print(f"  Accuracy: {backtest_results['over_under']['accuracy']:.1%}")
    print(f"  Estimated ROI: {backtest_results['over_under']['roi_estimate']:.1f}%")
    
    print("\n📊 BTTS Strategy:")
    print(f"  Total Bets: {backtest_results['btts']['total_bets']}")
    print(f"  Correct: {backtest_results['btts']['correct_bets']}")
    print(f"  Accuracy: {backtest_results['btts']['accuracy']:.1%}")
    print(f"  Estimated ROI: {backtest_results['btts']['roi_estimate']:.1f}%")
    
    print("\n" + "=" * 60)
    print("✓ Model trained on real Bundesliga data!")
    print("=" * 60)
    
    # ========================================================================
    # INTERACTIVE LIVE PREDICTIONS
    # ========================================================================
    
    print("\n\n" + "=" * 60)
    print(f"🔴 LIVE MATCH PREDICTIONS - {league_names[league_code].upper()}")
    print("=" * 60)
    
    # Show available teams
    print("\n📋 Available teams in database:")
    all_teams = sorted(set(raw_data['HomeTeam'].unique()) | set(raw_data['AwayTeam'].unique()))
    for i, team in enumerate(all_teams, 1):
        print(f"  {i:2d}. {team}")
    
    print("\n" + "=" * 60)
    print("Enter match details (or press Enter to skip)")
    print("=" * 60)
    
    while True:
        print("\n")
        home_team = input("Home team (or press Enter to finish): ").strip()
        
        if not home_team:
            print("\n✓ Exiting prediction mode")
            break
        
        away_team = input("Away team: ").strip()
        
        # Get match date
        match_date = input("Match date (YYYY-MM-DD, or press Enter for today): ").strip()
        if not match_date:
            from datetime import datetime
            match_date = datetime.now().strftime('%Y-%m-%d')
        
        # Validate teams exist in database
        if home_team not in all_teams:
            print(f"⚠️  Warning: '{home_team}' not found in database. Prediction may be inaccurate.")
            proceed = input("Continue anyway? (y/n): ").lower()
            if proceed != 'y':
                continue
        
        if away_team not in all_teams:
            print(f"⚠️  Warning: '{away_team}' not found in database. Prediction may be inaccurate.")
            proceed = input("Continue anyway? (y/n): ").lower()
            if proceed != 'y':
                continue
        
        # Get odds
        print("\nEnter odds in decimal format (e.g., 1.56):")
        try:
            over_odds = float(input("  Over 2.5 goals: "))
            under_odds = float(input("  Under 2.5 goals: "))
            btts_yes_odds = float(input("  BTTS Yes: "))
            btts_no_odds = float(input("  BTTS No: "))
            
            # Corners odds (optional)
            print("\nCorners odds (press Enter to skip):")
            over_corners_input = input("  Over 10.5 corners (or press Enter): ").strip()
            under_corners_input = input("  Under 10.5 corners (or press Enter): ").strip()
            
            over_corners_odds = float(over_corners_input) if over_corners_input else None
            under_corners_odds = float(under_corners_input) if under_corners_input else None
            
        except ValueError:
            print("❌ Invalid odds format. Skipping odds analysis.")
            over_odds = under_odds = btts_yes_odds = btts_no_odds = None
            over_corners_odds = under_corners_odds = None
        
        # Make prediction
        try:
            prediction = predict_upcoming_match(predictor, raw_data, home_team, away_team)
            
            # Value betting analysis
            if over_odds and under_odds and btts_yes_odds and btts_no_odds:
                print(f"\n{'='*60}")
                print("💰 VALUE BETTING ANALYSIS")
                print(f"{'='*60}")
                
                # Calculate expected value for each bet
                ou_prob = prediction['over_under']['hybrid_prob']
                ou_prediction = prediction['over_under']['prediction']
                
                if ou_prediction == "Over 2.5":
                    ou_ev = (ou_prob * over_odds) - 1
                    ou_bet_odds = over_odds
                    ou_bet_type = "Over 2.5"
                else:
                    ou_prob = 1 - ou_prob
                    ou_ev = (ou_prob * under_odds) - 1
                    ou_bet_odds = under_odds
                    ou_bet_type = "Under 2.5"
                
                btts_prob = prediction['btts']['hybrid_prob']
                btts_prediction = prediction['btts']['prediction']
                
                if btts_prediction == "Yes":
                    btts_ev = (btts_prob * btts_yes_odds) - 1
                    btts_bet_odds = btts_yes_odds
                    btts_bet_type = "BTTS Yes"
                else:
                    btts_prob = 1 - btts_prob
                    btts_ev = (btts_prob * btts_no_odds) - 1
                    btts_bet_odds = btts_no_odds
                    btts_bet_type = "BTTS No"
                
                # Display value bets
                print(f"\n📊 Over/Under 2.5:")
                print(f"  Recommended: {ou_bet_type}")
                print(f"  Model probability: {ou_prob:.1%}")
                print(f"  Bookmaker odds: {ou_bet_odds:.2f}")
                print(f"  Implied probability: {(1/ou_bet_odds):.1%}")
                print(f"  Expected Value: {ou_ev:+.1%}")
                
                if ou_ev > 0.05:  # 5% edge
                    print(f"  ✅ VALUE BET! Edge: {ou_ev:.1%}")
                elif ou_ev > 0:
                    print(f"  ⚠️  Slight edge: {ou_ev:.1%}")
                else:
                    print(f"  ❌ No value: {ou_ev:.1%}")
                
                print(f"\n📊 Both Teams To Score:")
                print(f"  Recommended: {btts_bet_type}")
                print(f"  Model probability: {btts_prob:.1%}")
                print(f"  Bookmaker odds: {btts_bet_odds:.2f}")
                print(f"  Implied probability: {(1/btts_bet_odds):.1%}")
                print(f"  Expected Value: {btts_ev:+.1%}")
                
                if btts_ev > 0.05:
                    print(f"  ✅ VALUE BET! Edge: {btts_ev:.1%}")
                elif btts_ev > 0:
                    print(f"  ⚠️  Slight edge: {btts_ev:.1%}")
                else:
                    print(f"  ❌ No value: {btts_ev:.1%}")
                
                # Corners value analysis
                if over_corners_odds and under_corners_odds:
                    print(f"\n📊 Corners (Over/Under 10.5):")
                    
                    corners_prob = corners_prediction['over_10_5']
                    
                    # Calculate EV for both sides
                    over_corners_ev = (corners_prob * over_corners_odds) - 1
                    under_corners_ev = ((1 - corners_prob) * under_corners_odds) - 1
                    
                    print(f"  Model probability (Over): {corners_prob:.1%}")
                    print(f"  Over 10.5 odds: {over_corners_odds:.2f} (Implied: {(1/over_corners_odds):.1%})")
                    print(f"  Over EV: {over_corners_ev:+.1%}")
                    
                    print(f"\n  Model probability (Under): {(1-corners_prob):.1%}")
                    print(f"  Under 10.5 odds: {under_corners_odds:.2f} (Implied: {(1/under_corners_odds):.1%})")
                    print(f"  Under EV: {under_corners_ev:+.1%}")
                    
                    if over_corners_ev > 0.05:
                        print(f"  ✅ VALUE BET! Over 10.5 corners (Edge: {over_corners_ev:.1%})")
                    elif under_corners_ev > 0.05:
                        print(f"  ✅ VALUE BET! Under 10.5 corners (Edge: {under_corners_ev:.1%})")
                    else:
                        print(f"  ❌ No strong value on corners")
                
                # Overall recommendation
                print(f"\n{'='*60}")
                print("🎯 FINAL RECOMMENDATION")
                print(f"{'='*60}")
                
                value_bets = []
                if ou_ev > 0.05 and prediction['over_under']['confidence'] >= 0.60:
                    value_bets.append(f"{ou_bet_type} @ {ou_bet_odds:.2f} (EV: {ou_ev:+.1%})")
                if btts_ev > 0.05 and prediction['btts']['confidence'] >= 0.60:
                    value_bets.append(f"{btts_bet_type} @ {btts_bet_odds:.2f} (EV: {btts_ev:+.1%})")
                
                # Add corners to recommendations
                if over_corners_odds and under_corners_odds:
                    if over_corners_ev > 0.05 and corners_prob >= 0.60:
                        value_bets.append(f"Over 10.5 corners @ {over_corners_odds:.2f} (EV: {over_corners_ev:+.1%})")
                    elif under_corners_ev > 0.05 and (1 - corners_prob) >= 0.60:
                        value_bets.append(f"Under 10.5 corners @ {under_corners_odds:.2f} (EV: {under_corners_ev:+.1%})")
                
                if value_bets:
                    print("\n✅ RECOMMENDED BETS:")
                    for bet in value_bets:
                        print(f"  • {bet}")
                else:
                    print("\n⚠️  No value bets found with sufficient edge and confidence")
                    print("  Consider skipping this match or waiting for better odds")
        
        except Exception as e:
            print(f"\n❌ Error making prediction: {e}")
            import traceback
            traceback.print_exc()
        
        # Ask if user wants to place/track a bet
        print("\n" + "=" * 60)
        track_bet = input("\n💰 Do you want to track a bet for this match? (y/n): ").lower()
        
        if track_bet == 'y':
            print("\nEnter bet details:")
            bet_type = input("  Bet type (e.g., 'Over 2.5', 'BTTS Yes', 'Over 10.5 corners'): ").strip()
            
            try:
                bet_odds = float(input("  Odds: "))
                bet_stake = float(input("  Stake (units): "))
                
                # Save to CSV
                save_bet_to_csv(
                    date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    bet_type=bet_type,
                    odds=bet_odds,
                    stake=bet_stake
                )
                
                print(f"\n📝 Bet tracked:")
                print(f"   Match: {home_team} vs {away_team}")
                print(f"   Date: {match_date}")
                print(f"   Bet: {bet_type} @ {bet_odds:.2f}")
                print(f"   Stake: {bet_stake} units")
                print(f"   Potential return: {bet_stake * bet_odds:.2f} units")
                
            except ValueError:
                print("❌ Invalid input. Bet not tracked.")
        
        # Ask if user wants to predict another match
        another = input("\nPredict another match? (y/n): ").lower()
        if another != 'y':
            break
    
    print("\n" + "=" * 60)
    print("✓ Prediction session complete!")
    print("=" * 60)
    
    # Option to view betting history
    view_history = input("\n📊 View betting history? (y/n): ").lower()
    if view_history == 'y':
        view_betting_history()
    
    print("\n💡 Tips:")
    print("  • Look for bets with EV > 5% and confidence > 60%")
    print("  • Bankroll management: Never bet more than 1-2% per bet")
    print("  • Track your results to refine the model over time")
    print("  • Remember: Past performance doesn't guarantee future results")
    print("=" * 60)

# ⚽ Football Betting Predictor

**AI-powered betting predictions using Machine Learning + Monte Carlo simulation**

Live Demo: https://footballpredict-260202.streamlit.app/

## 🎯 Features

### 📊 Multi-Market Predictions
- **Goals:** Over/Under 2.5 & Both Teams To Score (BTTS)
- **Corners:** Over/Under 10.5 corners
- **Cards:** Over/Under 4.5 total cards
- **Value Betting:** Expected Value (EV) analysis for each market

### 🌍 Supported Leagues
- 🇩🇪 Bundesliga (Germany)
- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship (England)
- 🇹🇷 Super Lig (Turkey)

### 🤖 Dual Prediction System
- **Machine Learning:** Gradient Boosting + Random Forest models
- **Monte Carlo:** 10,000 Poisson simulations per prediction
- **Hybrid:** Weighted combination for optimal accuracy

### 📈 Smart Analysis
- Recent form visualization (W/D/L with color coding)
- Fatigue analysis (days since last match)
- Team statistics (last 5 games)
- Head-to-head history
- Confidence-based recommendations

### 💰 Bet Tracking
- Save bets with date, odds, and stake
- Track results (Won/Lost/Pending)
- View betting history and statistics
- Export to Excel
- Calculate profit/loss and ROI

## 🚀 Quick Start

### Option 1: Web App (Easiest)
Just visit the live demo link above - works on any device!

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/football-betting-predictor.git
cd football-betting-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run ui.py
```

The app will open in your browser at `http://localhost:8501`

### Option 3: Command Line

```bash
python bundesliga_predictor.py
```

Interactive mode - just follow the prompts!

## 📱 Use on Mobile

The Streamlit web app works perfectly on mobile browsers:
- iOS Safari
- Android Chrome
- Any modern mobile browser

Just bookmark the URL for quick access!

## 🎓 How It Works

### 1. Data Collection
- Downloads historical match data from [football-data.co.uk](https://www.football-data.co.uk)
- Covers 4+ seasons (2021/22 to current)
- Automatically caches data locally
- Updates current season on each run

### 2. Feature Engineering
- Goals scored/conceded averages
- Recent form (last 3-5 games with decay)
- Home/away splits
- Head-to-head statistics
- Corners and cards averages
- Team volatility and consistency

### 3. Machine Learning
- **Over/Under model:** Gradient Boosting Classifier
- **BTTS model:** Random Forest Classifier
- Cross-validated for accuracy
- Trained on 1000+ matches

### 4. Monte Carlo Simulation
- Poisson distribution for goals, corners, and cards
- 10,000 iterations per prediction
- Probability distribution analysis
- Expected values calculation

### 5. Value Betting
- Compares model probability vs bookmaker odds
- Calculates Expected Value (EV)
- Only recommends bets with edge > 5% and confidence > 60%
- Helps find profitable opportunities

## 📊 Accuracy

Based on backtesting:
- **Over/Under 2.5:** ~65% accuracy, 24% ROI
- **BTTS:** ~60% accuracy, 15% ROI
- **Corners:** Often better value (less efficient market)
- **Cards:** Predictable based on team discipline

*Note: Past performance doesn't guarantee future results*

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **scikit-learn** - Machine learning models
- **NumPy/Pandas** - Data processing
- **OpenPyXL** - Excel export

## 📂 Project Structure

```
football-betting-predictor/
├── ui.py                          # Streamlit web interface
├── bundesliga_predictor.py        # Core prediction engine
├── betting_tracker_manager.py     # Bet management tool
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── data_cache/                    # Cached historical data (auto-created)
```

## 💡 Usage Tips

### Finding Value
- Look for bets with **EV > 5%** and **confidence > 60%**
- Markets with less bookmaker attention (corners, cards) often have more value
- Consider fatigue - teams playing with < 3 days rest

### Bankroll Management
- Never bet more than 1-2% of bankroll per bet
- Track all bets to measure long-term performance
- Focus on quality over quantity

### Best Practices
- Compare odds across multiple bookmakers
- Consider team news and injuries
- Weather can affect corners and cards
- Home advantage matters (especially for goals)

## 🔧 Advanced Usage

### Update Bet Results

```bash
python betting_tracker_manager.py
```

Menu options:
1. View betting history
2. Update bet results (Won/Lost)
3. Export to Excel
4. Exit

### Add More Leagues

Edit `bundesliga_predictor.py` line ~590:
```python
league_map = {
    '1': 'D1',  # Bundesliga
    '2': 'E1',  # Championship
    '3': 'T1',  # Super Lig
    '4': 'E0',  # Premier League (add this)
}
```

Available leagues: https://www.football-data.co.uk/data.php

### Customize Thresholds

In the UI sidebar or advanced settings:
- **ML/MC Weight:** Balance between models (default 60/40)
- **Confidence Threshold:** Minimum confidence to recommend (default 60%)
- **Edge Threshold:** Minimum expected value (default 5%)

## 📈 Future Enhancements

Potential improvements:
- [ ] More leagues (Premier League, La Liga, Serie A)
- [ ] Player statistics and lineups
- [ ] Weather data integration
- [ ] Live odds API integration
- [ ] Automated notifications for value bets
- [ ] Kelly Criterion bankroll management
- [ ] Referee tendencies for cards
- [ ] xG (Expected Goals) data

## ⚠️ Disclaimer

**For educational and entertainment purposes only.**

- This tool provides statistical predictions, not guarantees
- Gambling involves risk - never bet more than you can afford to lose
- Past performance does not indicate future results
- Always gamble responsibly
- Check your local laws regarding sports betting
- The authors are not responsible for any losses

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Share your results

## 📄 License

MIT License - feel free to use and modify for personal use.

## 🙏 Acknowledgments

- Data source: [football-data.co.uk](https://www.football-data.co.uk)
- Built with [Streamlit](https://streamlit.io)
- ML powered by [scikit-learn](https://scikit-learn.org)

## 📞 Support

Having issues? 
- Check the [Troubleshooting](#-troubleshooting) section
- Open an issue on GitHub
- Review the code comments for technical details

---

**Made with ⚽ and 🤖**

*Remember: The house always has an edge. Bet smart, bet small, bet for fun.*

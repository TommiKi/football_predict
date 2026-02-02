"""
Betting Tracker Manager
Update bet results and view history
"""

import pandas as pd
import os
from bundesliga_predictor import view_betting_history, update_bet_result

def main():
    csv_file = 'betting_tracker.csv'
    
    while True:
        print("\n" + "=" * 60)
        print("BETTING TRACKER MANAGER")
        print("=" * 60)
        print("\n1. View betting history")
        print("2. Update bet result")
        print("3. Export to Excel")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            view_betting_history(csv_file)
            
        elif choice == '2':
            if not os.path.isfile(csv_file):
                print("\n❌ No betting history found")
                continue
            
            df = pd.read_csv(csv_file)
            
            # Show pending bets
            pending = df[df['Result'] == 'Pending']
            
            if len(pending) == 0:
                print("\n✓ No pending bets")
                continue
            
            print("\n📋 PENDING BETS:")
            print("=" * 60)
            for idx, row in pending.iterrows():
                print(f"\n{idx}. {row['Match']} - {row['Date']}")
                print(f"   Bet: {row['Bet_Type']} @ {row['Odds']:.2f}")
                print(f"   Stake: {row['Stake']} units")
            
            try:
                bet_idx = int(input("\nEnter bet number to update: "))
                result = input("Result (Won/Lost): ").strip().capitalize()
                
                if result in ['Won', 'Lost']:
                    update_bet_result(csv_file, bet_idx, result)
                else:
                    print("❌ Invalid result. Use 'Won' or 'Lost'")
            except (ValueError, IndexError):
                print("❌ Invalid bet number")
        
        elif choice == '3':
            if not os.path.isfile(csv_file):
                print("\n❌ No betting history found")
                continue
            
            df = pd.read_csv(csv_file)
            excel_file = 'betting_tracker.xlsx'
            df.to_excel(excel_file, index=False)
            print(f"\n✅ Exported to {excel_file}")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid option")


if __name__ == "__main__":
    main()

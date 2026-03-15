import uuid
import time

class PaperBroker:
    def __init__(self, target_pts: float = 10.0, stop_loss_pts: float = 7.0, timeout_secs: int = 120):
        self.target_pts = target_pts
        self.stop_loss_pts = stop_loss_pts
        self.timeout_secs = timeout_secs
        
        self.position = 0 # 1=Long, -1=Short, 0=Flat
        self.active_trade = None # Only one active trade at a time for this strategy
        self.closed_trades = []
        self.trades_today = 0
        
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Stats
        self.total_execution_delay_ms = 0.0
        self.execution_count = 0

    def format_ist(self, timestamp: int):
        # Rough IST approximation (+5.5h)
        return time.strftime('%H:%M:%S', time.gmtime(timestamp + 19800))

    def execute(self, action_str: str, direction: str, price: float, timestamp: int):
        """
        Simulates order entry and returns the trade dictionary.
        Only allows one active trade at a time.
        """
        if action_str == "ENTER_NOW" and direction and self.active_trade is None:
            trade_id = str(uuid.uuid4())[:8]
            
            # Compute TP/SL Levels
            if direction == "BUY":
                target_price = price + self.target_pts
                stop_price = price - self.stop_loss_pts
                self.position = 1
            else:
                target_price = price - self.target_pts
                stop_price = price + self.stop_loss_pts
                self.position = -1
                
            trade = {
                "trade_id": trade_id,
                "direction": direction,
                "status": "OPEN",
                
                "entry_time": timestamp,
                "entry_time_ist": self.format_ist(timestamp),
                "entry_price": price,
                
                "current_price": price,
                "target_price": target_price,
                "stop_price": stop_price,
                
                "holding_time_seconds": 0,
                
                "exit_time": None,
                "exit_time_ist": None,
                "exit_price": None,
                "exit_reason": None,
                
                "pnl": 0.0
            }
            self.active_trade = trade
            self.trades_today += 1
            return trade
        return None

    def update_state(self, current_price: float, timestamp: int):
        """
        Updates active trade, checks for TP/SL/TIMEOUT.
        Returns the trade if it was closed in this step.
        """
        if not self.active_trade:
            self.unrealized_pnl = 0.0
            return None
            
        t = self.active_trade
        t["current_price"] = current_price
        t["holding_time_seconds"] = timestamp - t["entry_time"]
        
        # 1. Calc Unrealized
        if t['direction'] == "BUY":
            pnl = current_price - t['entry_price']
        else:
            pnl = t['entry_price'] - current_price
        
        self.unrealized_pnl = pnl
        t["pnl"] = pnl
        
        # 2. Check Exits (Priority: TP/SL > TIMEOUT)
        exit_reason = None
        
        # --- TP/SL CHECK ---
        if t['direction'] == "BUY":
            if current_price >= t['target_price']: 
                exit_reason = "TARGET"
            elif current_price <= t['stop_price']: 
                exit_reason = "STOP"
        else: # SELL
            if current_price <= t['target_price']: 
                exit_reason = "TARGET"
            elif current_price >= t['stop_price']: 
                exit_reason = "STOP"
            
        # --- TIMEOUT CHECK (Only if not already TP/SL) ---
        if not exit_reason and t["holding_time_seconds"] >= self.timeout_secs:
            exit_reason = "TIMEOUT"
            
        if exit_reason:
            t['status'] = "CLOSED"
            t['exit_reason'] = exit_reason
            t['exit_price'] = current_price
            t['exit_time'] = timestamp
            t['exit_time_ist'] = self.format_ist(timestamp)
            
            self.realized_pnl += pnl
            self.closed_trades.append(t)
            self.active_trade = None
            self.position = 0
            self.unrealized_pnl = 0.0
            return t
            
        return None

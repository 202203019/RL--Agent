import time
import os
from datetime import datetime
from queue import Queue
from ..bars.bar_types import Tick

class ReplayFeed:
    def __init__(self, file_path: str, tick_queue: Queue):
        self.file_path = file_path
        self.tick_queue = tick_queue
        self.first_tick = True

    def run(self, max_minutes: int = 10):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Replay file not found: {self.file_path}")

        print(f"Reading tick file: {self.file_path}")
        
        start_epoch = None
        current_minute = 0
        max_seconds = max_minutes * 60

        with open(self.file_path, 'r') as f:
            headers = f.readline().strip().split()
            # Assuming headers: epoch bid ask ltp open high low close ...
            try:
                idx_epoch = headers.index('epoch')
                idx_ltp = headers.index('ltp')
                idx_bid = headers.index('bid')
                idx_ask = headers.index('ask')
            except ValueError as e:
                # Fallback indices if header is missing or differs slightly
                idx_epoch, idx_bid, idx_ask, idx_ltp = 0, 1, 2, 3
            
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                try:
                    epoch = float(parts[idx_epoch])
                    ltp = float(parts[idx_ltp])
                    bid = float(parts[idx_bid])
                    ask = float(parts[idx_ask])
                except ValueError:
                    continue # Skip malformed lines
                
                if self.first_tick:
                    self.first_tick = False
                    start_epoch = epoch
                    ist_time = datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M:%S')
                    print("First tick detected:")
                    print(f"epoch: {int(epoch)}")
                    print(f"IST: {ist_time}")
                
                elapsed_seconds = epoch - start_epoch
                if elapsed_seconds > max_seconds:
                    break # Reached max replay time (e.g. 10 minutes)
                
                # Progress logging every 60 seconds
                if int(elapsed_seconds) // 60 > current_minute:
                    current_minute = int(elapsed_seconds) // 60
                    print(f"Replay Progress:\nProcessed: {current_minute * 60} seconds")

                self.tick_queue.put(Tick(epoch=epoch, price=ltp, bid=bid, ask=ask))

import os
from datetime import datetime

def write_logs(line):
    logs_date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    #print(logs_date)
    save_txt = 1
    if save_txt:  # Write to file
        with open(logs_date + '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
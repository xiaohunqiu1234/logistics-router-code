
from datetime import datetime, timedelta

import pandas as pd
delivery = pd.read_csv(r'D:\mcp\Artificial_intelligence\logistics-router-code\example_data\deliveries.csv')
time_window_start_dt = datetime.strptime(
                                    delivery['time_window_start'], '%H:%M'
                                )
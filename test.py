from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
print(formatted_time)
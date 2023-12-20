import requests

url = 'https://1y2z617mw1.execute-api.us-east-1.amazonaws.com/test/predict'

data = {'url': 'https://i.ibb.co/44t6ByP/20231220-105516.jpg'}

result = requests.post(url, json=data).json()
print(result)
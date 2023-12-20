import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://i.ibb.co/44t6ByP/20231220-105516.jpg'}

result = requests.post(url, json=data).json()
print(result)
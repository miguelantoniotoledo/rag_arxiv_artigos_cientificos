import requests

r = requests.get('https://ge.globo.com/busca?q=curling', timeout=15)
print('Status', r.status_code)
print(r.text[:4000])

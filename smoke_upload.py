import requests
import io
csv = 'a,b,c\n1,2,3\n4,5,6\n'
files = {'file': ('sample.csv', io.BytesIO(csv.encode()), 'text/csv')}
res = requests.post('http://127.0.0.1:8001/upload', files=files, timeout=30)
print('status', res.status_code)
try:
    data = res.json()
    import json
    print(json.dumps(data, indent=2))
except Exception as e:
    print('json error', e, res.text)

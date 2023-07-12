import urllib.request, json 

# Get sarcasm detection datset
with urllib.request.urlopen("https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json") as url:
    data = json.load(url)


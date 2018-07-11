#import libraries
import requests
import json

#set credentials
API_SECRET = "c024fcda4853206f3a00709ff529c9781af0f514"
#API_KEY = "453436d8627b58b401541282b7af14c6"

#get the url
url = "https://rest.viglink.com/api/product/search?apiKey=453436d8627b58b401541282b7af14c6&query=nike+running+shoes&country=us&category=Fashion&itemsPerPage=100"

#perform a GET request
response = requests.get(url, headers = {'Authorization': f'secret {API_SECRET}'})

#convert to json
response.json()
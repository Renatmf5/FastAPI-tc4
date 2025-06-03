# core/session.py
from curl_cffi import requests as curl_cffi_request

session = curl_cffi_request.Session()

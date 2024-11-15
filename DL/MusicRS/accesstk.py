import requests
import base64

def get_access():

    # Replace with your own Client ID and Client Secret
    CLIENT_ID = 'f192552c89d443dc92370edab281f66e'
    CLIENT_SECRET = 'e0a2e71027454d769e5f8a5514c624f5'

    # Base64 encode the client ID and client secret
    client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    client_credentials_base64 = base64.b64encode(client_credentials.encode())

    # Request the access token
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {client_credentials_base64.decode()}'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=data, headers=headers)

    if response.status_code == 200:
        access_token = response.json()['access_token']
        print("Access token obtained successfully.")
        return access_token
    else:
        print("Error obtaining access token.")
        exit()
import json

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from buaro import clms


class Reply:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_service_key_signs_a_valid_jwt_grant(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()).decode()
    key = {"client_id": "client", "user_id": "user", "token_uri": "https://land.copernicus.eu/@@oauth2-token", "private_key": pem, "key_id": "k"}
    monkeypatch.setenv("CLMS_SERVICE_KEY", json.dumps(key))
    seen = {}

    def fake_post(url, headers=None, data=None, timeout=None):
        seen["url"] = url
        seen["data"] = data
        return Reply({"access_token": "abc"})

    monkeypatch.setattr("requests.post", fake_post)
    token = clms.access_token(clms.service_key())
    assert token == "abc"
    assert seen["url"] == key["token_uri"]
    assert seen["data"]["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
    claims = jwt.decode(seen["data"]["assertion"], private.public_key(), algorithms=["RS256"], audience=key["token_uri"])
    assert claims["iss"] == "client" and claims["sub"] == "user"
    assert claims["exp"] - claims["iat"] == 3600

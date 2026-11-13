import json
import os
import time

import jwt
import requests

API = "https://land.copernicus.eu/api"


def service_key(env="CLMS_SERVICE_KEY"):
    raw = os.environ.get(env, "").strip()
    if not raw:
        raise RuntimeError(f"Environment variable {env} is empty.")
    return json.loads(raw)


def access_token(key, timeout=60):
    now = int(time.time())
    claims = {"iss": key["client_id"], "sub": key["user_id"], "aud": key["token_uri"], "iat": now, "exp": now + 3600}
    grant = jwt.encode(claims, key["private_key"].encode("utf-8"), algorithm="RS256")
    response = requests.post(
        key["token_uri"],
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": grant},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["access_token"]


class Client:
    def __init__(self, key=None, api=API):
        self.key = key
        self.api = api
        self.token = None

    def headers(self, auth):
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        if auth:
            if self.token is None:
                self.token = access_token(self.key)
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def request(self, method, path, auth=False, retry=True, **kwargs):
        url = path if path.startswith("http") else f"{self.api}/{path.lstrip('/')}"
        response = requests.request(method, url, headers=self.headers(auth), timeout=120, **kwargs)
        if auth and retry and response.status_code == 401:
            self.token = None
            return self.request(method, path, auth, False, **kwargs)
        return response

    def search(self, text, fields=("UID", "dataset_full_format", "dataset_download_information", "downloadable_files")):
        params = [("portal_type", "DataSet"), ("SearchableText", text), ("b_size", "50")]
        params += [("metadata_fields", f) for f in fields]
        response = self.request("GET", "@search", params=params)
        response.raise_for_status()
        return response.json().get("items", [])

    def get(self, path, auth=False):
        response = self.request("GET", path, auth=auth)
        response.raise_for_status()
        return response.json()

    def post_request(self, datasets):
        response = self.request("POST", "@datarequest_post", auth=True, data=json.dumps({"Datasets": datasets}))
        return response

    def status(self, task_id):
        response = self.request("GET", "@datarequest_status_get", auth=True, params={"TaskID": task_id})
        return response

    def my_requests(self):
        response = self.request("GET", "@datarequest_search", auth=True)
        return response

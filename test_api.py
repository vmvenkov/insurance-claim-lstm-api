import requests


BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health:", response.json())


def train_model():
    response = requests.post(f"{BASE_URL}/train")
    print("Train:", response.json())


def test_prediction():
    payload = {
        "description": "След силна градушка автомобилът има счупено предно стъкло и вдлъбнатини по капака."
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Prediction:", response.json())


def test_model_info():
    response = requests.get(f"{BASE_URL}/model-info")
    print("Model info:", response.json())


if __name__ == "__main__":
    test_health()
    train_model()
    test_model_info()
    test_prediction()

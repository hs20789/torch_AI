import requests
import json


def test_inference():
    # Test data
    test_input = 5.0

    # Make prediction request
    url = "http://localhost:8080/predictions/simple_linear"
    response = requests.post(url, data=str(test_input))

    print(f"Input: {test_input}")
    print(f"Prediction: {response.json()}")


if __name__ == "__main__":
    test_inference()

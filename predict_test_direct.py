import pandas as pd
import requests

# Change port value accordingly:
# 9696 for testing directly or with docker (with port mapping '9696:9696')
# 8080 for testing with kubernetes (with port-forwarding '8080:80')
url = "http://localhost:9696/predict"  # currently configured for kubernetes

# Load test dataset
test_data_file = "./test_data/test_data.csv"
test_data = pd.read_csv(test_data_file)
test_data_len = len(test_data)

# Greeting and first message
print()
print("Combined Cycle Power Plant Power Output Prediction")
print("―" * 50)
print()
print(
    f"Thank you for testing CCPP power output predictor. \
There are {test_data_len} test cases."
)

# Initilize instruction and user input value
message = f"Enter a number from 0 to {test_data_len-1} to select a test case \
(Enter '-1' to quit): "

user_input = 0

# Repeating test loop with user input for choice of test case
while user_input != -1:  # Exit loop if '-1' entered
    try:
        user_input = int(input(message))  # Get user input

        if user_input == -1:  # Skip if '-1' entered
            pass
        elif user_input >= 0 and user_input < test_data_len:  # Valid input
            test_case = dict(test_data.loc[user_input])

            PE_actual = test_case["PE"]
            del test_case["PE"]

            response = requests.post(url, json=test_case).json()
            print()
            print("Case details:")
            print(test_case)
            print()
            print("Response:")
            print(response)
            print()

            if response["PE"]:
                print(f"PE Predicted: {response['PE']}")
                print(f"PE Actual:    {PE_actual}")

            print()
        else:  # Invalid input
            print("Entered value is out of range.")
            print()

    except ValueError:
        print("Enter a valid integer.")
        print()
    except KeyError:
        print("Error: Invalid key in response.")
        print()
    except KeyboardInterrupt:
        print()
        print("Enter '-1' to quit")
        print()
    except requests.exceptions.RequestException as e:
        print()
        print(f"Request error: {e}")
        print()

print()
print("Test ended. Thank you!")
print()

import sys
import os
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.lambda_function import lambda_handler

class TestLambdaHandler(unittest.TestCase):
    def test_lambda_handler(self):
        event = {
            "body": json.dumps({"question": "What is composting?"})
        }
        context = {}
        response = lambda_handler(event, context)
        self.assertEqual(response['statusCode'], 200)
        self.assertIn('body', response)
        body = json.loads(response['body'])
        self.assertIn('query', body)
        self.assertIn('result', body)
        
        print("Request:")
        print(json.dumps(event, indent=2))
        
        print("\nResponse:")
        print(json.dumps(response, indent=2))
        

        print("\nChatbot Response:")
        print(body['result'])

if __name__ == '__main__':
    unittest.main()
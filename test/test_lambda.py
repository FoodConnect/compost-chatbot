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
        # Assert other properties of the response as needed
        self.assertIn('body', response)
        body = json.loads(response['body'])
        self.assertEqual(body['some_key'], 'expected_value')

if __name__ == '__main__':
    unittest.main()
import sys
import os
import unittest
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from vector_embeddings_handler.lambda_function import lambda_handler

class TestLambdaHandler(unittest.TestCase):
   def test_lambda_handler_success(self):
        event = {}
        context = {}
        response = lambda_handler(event, context)
        
        self.assertEqual(response['statusCode'], 200)
        self.assertIn('body', response)
        self.assertEqual(response['body'], 'Interaction successful!')

if __name__ == '__main__':
    unittest.main()